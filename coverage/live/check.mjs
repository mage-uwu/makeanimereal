#!/usr/bin/env node
// Check the browser runtime against the trained network: load coverage/live/ in headless Chromium
// (CDN modules served from node_modules), draw the reference frame's camera, and write
// <out>/browser.png next to the exporter's reference.png, plus a screenshot of the page.
//
//   node coverage/live/check.mjs [--out /tmp/check]
import { chromium } from 'playwright-core';
import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import { parseArgs } from 'node:util';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '../..');
const live = path.join(root, 'coverage/live');
const { values: opt } = parseArgs({ options: { out: { type: 'string', default: '/tmp/check' }, deep: { type: 'boolean', default: false }, poses: { type: 'boolean', default: false } } });
await fs.mkdir(opt.out, { recursive: true });

const types = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.png': 'image/png' };
const server = http.createServer(async (req, res) => {
  const p = new URL(req.url, 'http://x').pathname;
  try {
    const file = path.join(live, p === '/' ? 'index.html' : p);
    res.writeHead(200, { 'content-type': types[path.extname(file)] || 'application/octet-stream' }).end(await fs.readFile(file));
  } catch { res.writeHead(404).end(); }
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
  args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
});
try {
  const page = await browser.newPage({ viewport: { width: 900, height: 900 } });
  page.on('pageerror', e => console.error('[page]', e.message));
  page.on('console', m => { if (m.type() === 'error') console.error('[console]', m.text()); });
  await page.route('https://cdn.jsdelivr.net/npm/**', async route => {
    const u = new URL(route.request().url()).pathname.replace('/npm/', '');
    const [, name, rest] = /^(@[^/]+\/[^@]+|[^@/]+)@[^/]+\/(.*)$/.exec(u);
    route.fulfill({ path: path.join(root, 'node_modules', name, rest), contentType: 'text/javascript' });
  });
  await page.route('https://fonts.**', r => r.fulfill({ body: '', contentType: 'text/css' }));
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  await page.waitForFunction(() => window.ready || document.getElementById('status').textContent.startsWith('Could'), null, { timeout: 180000 });
  const ref = JSON.parse(await fs.readFile(path.join(live, 'reference.json'), 'utf8')).frame;
  const t0 = Date.now();
  const url = await page.evaluate(([y, p]) => window.debugFrame(y, p), [ref.yaw, ref.pitch]);
  console.log(`frame ${ref.frame} (yaw ${ref.yaw}, pitch ${ref.pitch}) drawn in ${Date.now() - t0} ms (software GL)`);
  await fs.writeFile(path.join(opt.out, 'browser.png'), Buffer.from(url.split(',')[1], 'base64'));
  // With --deep: poses the shader never saw (for a look by eye), the input tensor and intermediates.
  if (opt.deep && opt.poses) for (const [name, y, p] of [['wave', 20, 5], ['look', 330, 10]]) {
    const u = await page.evaluate(([n, y, p]) => window.debugFrame(y, p, n, 0.3), [name, y, p]);
    await fs.writeFile(path.join(opt.out, `pose-${name}.png`), Buffer.from(u.split(',')[1], 'base64'));
  }
  // The packed input tensor, for comparing with the training data's frame.
  const x = [];
  if (opt.deep) for (let k = 0; k < 5; k++) x.push(await page.evaluate(k => window.debugTensor('x', k), k));
  await fs.writeFile(path.join(opt.out, 'x.json'), JSON.stringify(x));
  // First layer of each intermediate tensor, to find where the browser and torch part ways.
  const inter = {};
  if (opt.deep) for (const n of ['d0_mid', 'd0', 'd0_pool', 'd1', 'd2', 'd3', 'u0', 'u1', 'u2', 'rgba']) inter[n] = await page.evaluate(n => window.debugTensor(n, 0), n);
  await fs.writeFile(path.join(opt.out, 'inter.json'), JSON.stringify(inter));
  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join(opt.out, 'page.png') });
} finally {
  await browser.close();
  server.close();
}
