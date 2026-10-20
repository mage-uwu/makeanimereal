#!/usr/bin/env node
// Render the GLB from a ring of camera angles (and optionally animation frames).
// Writes PNGs + manifest.json into the output directory.
//
//   node scripts/render.mjs [--model assets/nya-bytetuber.glb] [--out out/renders]
//        [--yaw-step 30] [--pitches -15,0,20] [--size 768x1152] [--fov 22]
//        [--background "#ffffff"] [--anim Idle] [--frames 1]
import { chromium } from 'playwright-core';
import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import { parseArgs } from 'node:util';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');

const { values: opt } = parseArgs({
  options: {
    model: { type: 'string', default: 'assets/nya-bytetuber.glb' },
    out: { type: 'string', default: 'out/renders' },
    'yaw-step': { type: 'string', default: '30' },
    pitches: { type: 'string', default: '-15,0,20' },
    size: { type: 'string', default: '768x1152' },
    fov: { type: 'string', default: '22' },
    background: { type: 'string', default: '#ffffff' },
    anim: { type: 'string' },
    frames: { type: 'string', default: '1' },
  },
});

const [width, height] = opt.size.split('x').map(Number);
const yawStep = Number(opt['yaw-step']);
const pitches = opt.pitches.split(',').map(Number);
const frames = Math.max(1, Number(opt.frames));
const outDir = path.resolve(opt.out);

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.glb': 'model/gltf-binary' };
const server = http.createServer(async (req, res) => {
  const file = path.join(root, decodeURIComponent(new URL(req.url, 'http://x').pathname));
  if (!file.startsWith(root)) return res.writeHead(403).end();
  try {
    const body = await fs.readFile(file);
    res.writeHead(200, { 'content-type': MIME[path.extname(file)] || 'application/octet-stream' }).end(body);
  } catch {
    res.writeHead(404).end();
  }
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;

const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || undefined,
  args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
});
try {
  const page = await browser.newPage({ viewport: { width, height } });
  page.on('pageerror', e => console.error('[page]', e.message));
  await page.goto(`${base}/scripts/render-page.html`);
  await page.waitForFunction(() => window.setup);
  const info = await page.evaluate(a => window.setup(a), {
    model: `/${path.relative(root, path.resolve(opt.model))}`,
    width, height, fov: Number(opt.fov), background: opt.background === 'none' ? null : opt.background,
  });
  console.log('animations:', info.clips.map(c => `${c.name} (${c.duration.toFixed(2)}s)`).join(', '));

  const clip = opt.anim ? info.clips.find(c => c.name === opt.anim) : null;
  if (opt.anim && !clip) throw new Error(`unknown animation ${opt.anim}`);

  await fs.mkdir(outDir, { recursive: true });
  const views = [];
  for (let f = 0; f < frames; f++) {
    const time = clip ? (clip.duration * f) / frames : 0;
    await page.evaluate(a => window.pose(a), { anim: opt.anim, time });
    for (const pitch of pitches) {
      for (let yaw = 0; yaw < 360; yaw += yawStep) {
        const dataUrl = await page.evaluate(a => window.shot(a), { yaw, pitch });
        const name = `${clip ? `${clip.name}_f${String(f).padStart(2, '0')}_` : ''}y${String(yaw).padStart(3, '0')}_p${pitch}.png`;
        await fs.writeFile(path.join(outDir, name), Buffer.from(dataUrl.split(',')[1], 'base64'));
        views.push({ file: name, yaw, pitch, frame: f, time, anim: opt.anim || null });
        process.stdout.write(`\r${views.length} views`);
      }
    }
  }
  process.stdout.write('\n');
  await fs.writeFile(path.join(outDir, 'manifest.json'), JSON.stringify({
    model: opt.model, width, height, fov: Number(opt.fov), yawStep, pitches, frames,
    anim: opt.anim || null, background: opt.background, views,
  }, null, 2));
  console.log(`wrote ${views.length} renders to ${path.relative(process.cwd(), outDir)}`);
} finally {
  await browser.close();
  server.close();
}
