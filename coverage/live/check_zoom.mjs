// The shader's input (G-buffer tiles) and output must not depend on devicePixelRatio (browser zoom):
// capture at DPR 1, change it mid-session (as zooming does), capture again, and load fresh at others.
//   node check_zoom.mjs index.html cpu debugCpu        (viewer, nanotss CPU)
//   WGPU=1 node check_zoom.mjs index.html ntsgpu debugCpu   (viewer, nanotss WebGPU)
//   node check_zoom.mjs index.html fast debugFrame     (viewer, learned shader on WebGL; WGPU=1 crisp: WebGPU)
//   node check_zoom.mjs cpu.html noloop debugGpu       (Tree-SSM page, GPU engine)
// Each line prints hashes of the output and tiles: they must all match. SETTLE: ms to let the first frame finish.
import { chromium } from 'playwright-core';
import fs from 'node:fs'; import path from 'node:path'; import crypto from 'node:crypto';
const WEB = new URL('.', import.meta.url).pathname;
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.png': 'image/png', '.wasm': 'application/wasm' };
const [page_, hash, hook] = process.argv.slice(2);   // e.g. index.html ntsgpu debugCpu
const gpuArgs = process.env.WGPU ? ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-features=Vulkan', '--use-vulkan=swiftshader'] : [];
const b = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome', args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', ...gpuArgs] });
const H = s => crypto.createHash('sha1').update(s).digest('hex').slice(0, 12);
async function open(dpr) {
  const p = await b.newPage({ viewport: { width: 900, height: 1000 }, deviceScaleFactor: dpr });
  await p.route('https://pixie.test/**', r => { const rel = new URL(r.request().url()).pathname.slice(1) || 'index.html'; const f = path.join(WEB, rel); if (!fs.existsSync(f)) return r.fulfill({ status: 404, body: '' }); r.fulfill({ status: 200, body: fs.readFileSync(f), contentType: TYPES[path.extname(f)] || 'application/octet-stream' }); });
  p.on('pageerror', e => console.log('[pageerror]', e.message));
  await p.goto(`https://pixie.test/${page_}#${hash}`, { timeout: 600000 });
  await p.waitForFunction(h => window.ready && (h.includes('noloop') || window.framesDrawn >= 1), hash, { timeout: 600000 }); await p.waitForTimeout(+(process.env.SETTLE || 15000));
  return p;
}
const grab = async p => {
  const r = await p.evaluate(h => window[h](20, -10, 'stand', 0), hook);
  return typeof r === 'string' ? H(r) : r.px ? `out ${H(r.px)} tiles ${H(r.tiles.join(''))}` : `out ${H(r.rgba)} tiles ${H(r.tiles)}`;
};
const p1 = await open(1);
console.log(`${page_}#${hash} ${hook}: DPR 1            ${await grab(p1)}  (devicePixelRatio ${await p1.evaluate(() => devicePixelRatio)})`);
const cdp = await p1.context().newCDPSession(p1);
await cdp.send('Emulation.setDeviceMetricsOverride', { width: 900, height: 1000, deviceScaleFactor: 1.5, mobile: false });
await p1.waitForTimeout(500);
console.log(`${page_}#${hash} ${hook}: DPR 1 -> 1.5 live  ${await grab(p1)}  (devicePixelRatio ${await p1.evaluate(() => devicePixelRatio)})`);
for (const d of (process.env.FRESH || '2').split(',').map(Number)) {
  const p2 = await open(d);
  console.log(`${page_}#${hash} ${hook}: DPR ${d} fresh      ${await grab(p2)}  (devicePixelRatio ${await p2.evaluate(() => devicePixelRatio)})`);
  await p2.close();
}
await b.close();
