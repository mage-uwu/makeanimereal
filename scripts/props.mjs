#!/usr/bin/env node
// Render the ground props' 3D proxies from a ring of angles (see scripts/props-page.html).
//
//   node scripts/props.mjs [--out out/props-renders] [--yaws 8] [--pitches 10,50,75]
import { chromium } from 'playwright-core';
import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import { parseArgs } from 'node:util';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const { values: opt } = parseArgs({ options: {
  out: { type: 'string', default: 'out/props-renders' },
  yaws: { type: 'string', default: '8' },
  pitches: { type: 'string', default: '10,50,75' },
} });
const PROPS = { rock: Number(opt.yaws), bush: Number(opt.yaws), grass: Math.max(1, Number(opt.yaws) / 2) };
const pitches = opt.pitches.split(',').map(Number);
const MIME = { '.html': 'text/html', '.js': 'text/javascript' };
const server = http.createServer(async (req, res) => {
  const file = path.join(root, decodeURIComponent(new URL(req.url, 'http://x').pathname));
  let body;
  try { body = await fs.readFile(file); } catch { return res.writeHead(404).end(); }
  res.writeHead(200, { 'content-type': MIME[path.extname(file)] || 'application/octet-stream' }).end(body);
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || undefined,
  args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
});
try {
  const page = await browser.newPage({ viewport: { width: 512, height: 512 } });
  page.on('pageerror', e => console.error('[page]', e.message));
  await page.goto(`http://127.0.0.1:${server.address().port}/scripts/props-page.html`);
  await page.waitForFunction(() => window.setup);
  const manifest = { props: {} };
  for (const [name, n] of Object.entries(PROPS)) {
    const info = await page.evaluate(nm => window.setup(nm), name);
    const yaws = Array.from({ length: n }, (_, i) => +(i * 360 / n).toFixed(2)), views = [];
    await fs.mkdir(path.join(opt.out, name), { recursive: true });
    for (const pitch of pitches) for (const yaw of yaws) {
      const url = await page.evaluate(a => window.shot(a), { yaw, pitch });
      const file = `${name}/y${yaw}_p${pitch}.png`;
      await fs.writeFile(path.join(opt.out, file), Buffer.from(url.split(',')[1], 'base64'));
      views.push({ file, yaw, pitch });
    }
    manifest.props[name] = { ...info, yaws, pitches, views };
    console.log(`${name}: ${views.length} views, frame ${info.frame.toFixed(2)} m`);
  }
  await fs.writeFile(path.join(opt.out, 'manifest.json'), JSON.stringify(manifest, null, 1));
} finally {
  await browser.close();
  server.close();
}
