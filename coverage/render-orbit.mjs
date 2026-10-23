#!/usr/bin/env node
// Render the total-coverage film: one camera spiral around the static model that passes every
// side at every height, as numbered PNGs (colour and depth), MP4s and cameras.json (the exact camera of each frame).
//
//   node coverage/render-orbit.mjs [--model coverage/assets/magical_girl_02.vrm] [--out out/coverage]
//        [--seconds 60] [--fps 16] [--turns 4] [--pitch=-20:70] [--size 720x1280] [--fov 30]
//        [--background "#00b140"] [--only 0,240,480]   (render just these frames, for a preview)
//        [--yaw0 0] [--skip-first] [--shorts] [--gbuffer]   (continue another spiral: start at its end yaw, without repeating that frame)
//        [--poses poses.json]   (instead of the spiral: [{clip, t, yaw, pitch[, light]}, ...] per frame, posed with
//                                the mocap clips in coverage/live/motions.json; clip 'stand' is the plain stand;
//                                light is {dir, color, intensity, amb, ambColor} for orbit-page's window.light)
//
// The spiral climbs linearly in pitch while turning at a constant yaw rate, so consecutive turns
// sit (pitch range / turns) apart and every frame moves the same small angle from the last.
import { chromium } from 'playwright-core';
import { spawnSync } from 'node:child_process';
import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import { parseArgs } from 'node:util';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const { values: opt } = parseArgs({
  options: {
    model: { type: 'string', default: 'coverage/assets/magical_girl_02.vrm' },
    out: { type: 'string', default: 'out/coverage' },
    seconds: { type: 'string', default: '60' },
    fps: { type: 'string', default: '16' },
    turns: { type: 'string', default: '4' }, // negative turns go the other way round
    yaw0: { type: 'string', default: '0' },
    'skip-first': { type: 'boolean', default: false },
    gbuffer: { type: 'boolean', default: false }, // also write normal/, uva/, uvb/ for the learned shader
    shorts: { type: 'boolean', default: false }, // navy safety shorts under the dress, for views from below
    pitch: { type: 'string', default: '-20:70' },
    size: { type: 'string', default: '720x1280' },
    fov: { type: 'string', default: '30' },
    background: { type: 'string', default: '#00b140' },
    'arm-drop': { type: 'string', default: '68' },
    only: { type: 'string' },
    poses: { type: 'string' },
  },
});
const [width, height] = opt.size.split('x').map(Number);
const [p0, p1] = opt.pitch.split(':').map(Number);
const fps = Number(opt.fps), turns = Number(opt.turns);
const skip = opt['skip-first'] ? 1 : 0, yaw0 = Number(opt.yaw0);
const poses = opt.poses ? JSON.parse(await fs.readFile(opt.poses, 'utf8')) : null;
const n = poses ? poses.length : Math.round(Number(opt.seconds) * fps);
const cam = i => poses ? { yaw: poses[i].yaw, pitch: poses[i].pitch } : spiral(i);
const spiral = i => { const t = (i + skip) / (n - 1 + skip); return { yaw: +(yaw0 + t * turns * 360).toFixed(4), pitch: +(p0 + t * (p1 - p0)).toFixed(4) }; };
const outDir = path.resolve(opt.out);
for (const d of ['frames', 'depth', ...(opt.gbuffer ? ['normal', 'uva', 'uvb'] : [])]) await fs.mkdir(path.join(outDir, d), { recursive: true });

const server = http.createServer(async (req, res) => {
  const file = path.join(root, decodeURIComponent(new URL(req.url, 'http://x').pathname));
  if (!file.startsWith(root)) return res.writeHead(403).end();
  try {
    const body = await fs.readFile(file);
    res.writeHead(200, { 'content-type': file.endsWith('.html') ? 'text/html' : file.endsWith('.js') ? 'text/javascript' : 'application/octet-stream' }).end(body);
  } catch { res.writeHead(404).end(); }
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || undefined,
  args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
});
try {
  const page = await browser.newPage({ viewport: { width, height } });
  page.on('pageerror', e => console.error('[page]', e.message));
  await page.goto(`http://127.0.0.1:${server.address().port}/coverage/orbit-page.html`);
  await page.waitForFunction(() => window.setup);
  const info = await page.evaluate(a => window.setup(a), {
    model: `/${path.relative(root, path.resolve(opt.model))}`, width, height,
    fov: Number(opt.fov), background: opt.background, armDrop: Number(opt['arm-drop']), shorts: opt.shorts,
    motions: poses ? '/coverage/live/motions.json' : null,
  });
  console.log('model', info);
  const which = opt.only ? opt.only.split(',').map(Number) : [...Array(n).keys()];
  const cameras = [];
  let parts = null;
  for (const i of which) {
    const c = cam(i);
    if (poses) await page.evaluate(([clip, t, light]) => { window.pose(clip, t); window.light(light || undefined); }, [poses[i].clip, poses[i].t, poses[i].light || null]);
    const s = await page.evaluate(([y, p]) => window.shot(y, p), [c.yaw, c.pitch]);
    const name = `${String(i).padStart(4, '0')}.png`;
    await fs.writeFile(path.join(outDir, 'frames', name), Buffer.from(s.png.split(',')[1], 'base64'));
    await fs.writeFile(path.join(outDir, 'depth', name), Buffer.from(s.depth.split(',')[1], 'base64'));
    if (opt.gbuffer) {
      const g = await page.evaluate(() => window.gbuffer());
      for (const k of ['normal', 'uva', 'uvb']) await fs.writeFile(path.join(outDir, k, name), Buffer.from(g[k].split(',')[1], 'base64'));
      parts = g.parts;
    }
    cameras.push({ frame: i, time: +(i / fps).toFixed(4), ...c, ...(poses ? { clip: poses[i].clip, t: poses[i].t, light: poses[i].light } : {}), position: s.position, quaternion: s.quaternion });
    if (i % 50 === 0) console.log(`frame ${i}/${n}  yaw ${c.yaw}  pitch ${c.pitch}`);
  }
  await fs.writeFile(path.join(outDir, 'cameras.json'), JSON.stringify({
    fps, frames: n, width, height, fov: Number(opt.fov), turns, pitch: [p0, p1],
    target: info.target, radius: info.radius, background: opt.background, parts, cameras,
  }));
  if (!opt.only && !poses) {
    for (const [dir, file] of [['frames', 'orbit.mp4'], ['depth', 'depth.mp4']])
      spawnSync('ffmpeg', ['-y', '-loglevel', 'error', '-framerate', String(fps), '-i', path.join(outDir, dir, '%04d.png'),
        '-c:v', 'libx264', '-crf', '14', '-pix_fmt', 'yuv420p', path.join(outDir, file)], { stdio: 'inherit' });
    console.log(`${n} frames -> ${outDir}/orbit.mp4, depth.mp4`);
  }
} finally {
  await browser.close();
  server.close();
}
