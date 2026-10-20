#!/usr/bin/env node
// Render the GLB from a ring of camera angles (and optionally animation frames).
// Writes PNGs + manifest.json into the output directory.
//
//   node scripts/render.mjs [--model assets/emerald-pixie.glb] [--out out/renders]
//        [--yaw-step 30] [--pitches=-15,0,20] [--size 768x1152] [--fov 22]
//        [--rings=-30:24,0:24,60:12]   (pitch:keys-per-ring; replaces --yaw-step/--pitches)
//        [--background "#ffffff"] [--anim Walking] [--frames 1] [--pose rest|relaxed]
//        [--export-mesh out/relaxed-renders/mesh.json] [--depth]
//        [--camera-from out/g2-renders/manifest.json]   (reuse that set's orbit for every frame)
import { chromium } from 'playwright-core';
import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import { parseArgs } from 'node:util';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');

const { values: opt } = parseArgs({
  options: {
    model: { type: 'string', default: 'assets/emerald-pixie.glb' },
    out: { type: 'string', default: 'out/renders' },
    'yaw-step': { type: 'string', default: '30' },
    pitches: { type: 'string', default: '-15,0,20' },
    rings: { type: 'string' },
    size: { type: 'string', default: '768x1152' },
    fov: { type: 'string', default: '22' },
    background: { type: 'string', default: '#ffffff' },
    anim: { type: 'string' },
    pose: { type: 'string', default: 'rest' }, // rest (the model's own pose) | relaxed (arms down)
    'export-mesh': { type: 'string' }, // also write the posed mesh as JSON (first frame)
    depth: { type: 'boolean', default: false }, // also write a depth map per view into <out>/depth
    frames: { type: 'string', default: '1' },
    'camera-from': { type: 'string' }, // pin the orbit to another render set's (frame 0) camera
  },
});

const [width, height] = opt.size.split('x').map(Number);
const yawStep = Number(opt['yaw-step']);
// Camera rings around the view sphere: each a pitch and its yaws. Rings near the top need fewer
// keys for the same spacing, so --rings sets the count per ring.
const rings = opt.rings
  ? opt.rings.split(',').map(r => { const [p, n] = r.split(':').map(Number); return { pitch: p, yaws: Array.from({ length: n }, (_, i) => +(i * 360 / n).toFixed(3)) }; })
  : opt.pitches.split(',').map(Number).map(p => ({ pitch: p, yaws: Array.from({ length: Math.round(360 / yawStep) }, (_, i) => i * yawStep) }));
const pitches = rings.map(r => r.pitch);
// Depth maps cover the character only, so tight planes around the orbit keep their precision.
const DEPTH_PAD = 2.2;
const depthRange = radius => ({ near: radius - DEPTH_PAD, far: radius + DEPTH_PAD });
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

  if (opt['camera-from']) {
    const other = JSON.parse(await fs.readFile(path.resolve(opt['camera-from']), 'utf8'));
    await page.evaluate(c => window.pinCamera(c), other.cameras[0]);
  }
  const clip = opt.anim ? info.clips.find(c => c.name === opt.anim) : null;
  if (opt.anim && !clip) throw new Error(`unknown animation ${opt.anim}`);

  await fs.mkdir(outDir, { recursive: true });
  const views = [];
  // Orbit centre and distance per frame, so a viewer can rebuild each keyframe's exact camera.
  const cameras = [];
  for (let f = 0; f < frames; f++) {
    const time = clip ? (clip.duration * f) / frames : 0;
    const cam = await page.evaluate(a => window.pose(a), { anim: opt.anim, time, pose: opt.pose });
    cameras.push({ frame: f, target: cam.center, radius: cam.radius });
    if (f === 0 && opt['export-mesh']) {
      const mesh = await page.evaluate(() => window.exportMesh());
      await fs.writeFile(path.resolve(opt['export-mesh']), JSON.stringify(mesh));
      console.log(`mesh: ${mesh.position.length / 3} vertices, ${mesh.index.length / 3} triangles -> ${opt['export-mesh']}`);
    }
    for (const { pitch, yaws } of rings) {
      for (const yaw of yaws) {
        const dataUrl = await page.evaluate(a => window.shot(a), { yaw, pitch });
        const name = `${clip ? `${clip.name}_f${String(f).padStart(2, '0')}_` : ''}y${String(yaw).padStart(3, '0')}_p${pitch}.png`;
        await fs.writeFile(path.join(outDir, name), Buffer.from(dataUrl.split(',')[1], 'base64'));
        if (opt.depth) {
          const url = await page.evaluate(a => window.shot(a), { yaw, pitch, depth: depthRange(cam.radius) });
          await fs.mkdir(path.join(outDir, 'depth'), { recursive: true });
          await fs.writeFile(path.join(outDir, 'depth', name), Buffer.from(url.split(',')[1], 'base64'));
        }
        views.push({ file: name, yaw, pitch, frame: f, time, anim: opt.anim || null });
        process.stdout.write(`\r${views.length} views`);
      }
    }
  }
  process.stdout.write('\n');
  await fs.writeFile(path.join(outDir, 'manifest.json'), JSON.stringify({
    model: opt.model, width, height, fov: Number(opt.fov), yawStep, pitches, frames,
    anim: opt.anim || null, pose: opt.pose, background: opt.background, cameras, views,
    depth: opt.depth ? { dir: 'depth', pad: DEPTH_PAD, packing: 'three RGBDepthPacking of gl_FragCoord.z' } : null,
  }, null, 2));
  console.log(`wrote ${views.length} renders to ${path.relative(process.cwd(), outDir)}`);
} finally {
  await browser.close();
  server.close();
}
