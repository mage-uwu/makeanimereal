#!/usr/bin/env node
// Training data for the learned render: her mesh with uv and base colour (into <renders>/mesh.json
// and <renders>/albedo.png), the feature grid's bounds, and a G-buffer per key camera, drawn by
// web/learned.js exactly as the viewer draws it.
//
//   node scripts/learned-data.mjs [--renders out/g2-renders] [--out out/learned]
import { chromium } from 'playwright-core';
import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import zlib from 'node:zlib';
import { parseArgs } from 'node:util';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const { values: opt } = parseArgs({ options: {
  renders: { type: 'string', default: 'out/g2-renders' },
  out: { type: 'string', default: 'out/learned' },
} });
const renders = path.resolve(opt.renders), out = path.resolve(opt.out);
const manifest = JSON.parse(await fs.readFile(path.join(renders, 'manifest.json'), 'utf8'));
const rel = f => `/${path.relative(root, f)}`;

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.png': 'image/png' };
const server = http.createServer(async (req, res) => {
  const file = path.join(root, decodeURIComponent(new URL(req.url, 'http://x').pathname));
  if (!file.startsWith(root)) return res.writeHead(403).end();
  let body;
  try { body = await fs.readFile(file); } catch { return res.writeHead(404).end(); }
  res.writeHead(200, { 'content-type': MIME[path.extname(file)] || 'application/octet-stream' }).end(body);
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || undefined,
  args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
});
try {
  // 1. The mesh in the keys' pose, now with uv, and its base colour texture.
  const page = await browser.newPage();
  page.on('pageerror', e => console.error('[page]', e.message));
  await page.goto(`${base}/scripts/render-page.html`);
  await page.waitForFunction(() => window.setup);
  await page.evaluate(a => window.setup(a), { model: rel(path.resolve(root, manifest.model)), width: 64, height: 96, fov: manifest.fov, background: null });
  await page.evaluate(a => window.pose(a), { pose: manifest.pose });
  const mesh = await page.evaluate(() => window.exportMesh());
  const old = JSON.parse(await fs.readFile(path.join(renders, 'mesh.json'), 'utf8'));
  const drift = Math.max(...old.position.map((v, i) => Math.abs(v - mesh.position[i])));
  if (old.position.length !== mesh.position.length || drift > 1e-3) throw new Error(`mesh differs from the keys' mesh (max drift ${drift})`);
  await fs.writeFile(path.join(renders, 'mesh.json'), JSON.stringify({ ...old, uv: mesh.uv }));
  const albedo = await page.evaluate(() => window.albedo());
  await fs.writeFile(path.join(renders, 'albedo.png'), Buffer.from(albedo.split(',')[1], 'base64'));
  await page.close();

  // 2. The feature grid's box: her bounds, padded a little.
  const lo = [0, 1, 2].map(a => Math.min(...mesh.position.filter((_, i) => i % 3 === a)));
  const hi = [0, 1, 2].map(a => Math.max(...mesh.position.filter((_, i) => i % 3 === a)));
  const box = { min: lo.map((v, a) => +(v - 0.03 * (hi[a] - lo[a]) - 0.01).toFixed(4)), size: hi.map((v, a) => +((v - lo[a]) * 1.06 + 0.02).toFixed(4)) };
  await fs.mkdir(path.join(out, 'gbuffer'), { recursive: true });
  await fs.writeFile(path.join(out, 'box.json'), JSON.stringify(box));

  // 3. A G-buffer per key camera.
  const g = await browser.newPage();
  g.on('pageerror', e => console.error('[page]', e.message));
  await g.goto(`${base}/scripts/gbuffer-page.html`);
  await g.waitForFunction(() => window.setup);
  const floatOk = await g.evaluate(a => window.setup(a), {
    mesh: rel(path.join(renders, 'mesh.json')), albedo: rel(path.join(renders, 'albedo.png')), box,
    fov: manifest.fov, aspect: manifest.width / manifest.height, camera: manifest.cameras[0],
  });
  if (!floatOk) throw new Error('no float render targets');
  let n = 0;
  for (const v of manifest.views) {
    const b64 = await g.evaluate(a => window.shot(a), { yaw: v.yaw, pitch: v.pitch });
    await fs.writeFile(path.join(out, 'gbuffer', v.file.replace('.png', '.u16.gz')), zlib.gzipSync(Buffer.from(b64, 'base64'), { level: 6 }));
    process.stdout.write(`\r${++n}/${manifest.views.length} G-buffers`);
  }
  process.stdout.write('\n');
} finally {
  await browser.close();
  server.close();
}
