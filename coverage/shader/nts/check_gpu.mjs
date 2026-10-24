// gpu-nts.js (WebGPU compute) against shader_nts.c (--ref) and infer.wasm, on one frame: the same tiles
// as check_infer.mjs builds, run in headless Chromium (WebGPU on SwiftShader where there's no GPU).
//   node check_gpu.mjs ref.bin fine.raw [net folder under coverage/live, default nts/]
// Prints check_infer.mjs's numbers (over white: largest and mean difference, values off by > 2/255)
// for WebGPU against the C trainer and against infer.wasm's bytes, and the frame time.
import { readFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright-core';

const [refPath, finePath, netDir = 'nts/'] = process.argv.slice(2);
const live = new URL('../../live/', import.meta.url).pathname;
const ref = readFileSync(refPath), dv = new DataView(ref.buffer, ref.byteOffset);
const H = dv.getInt32(4, true), W = dv.getInt32(8, true), pitch = dv.getFloat32(12, true) * Math.PI / 180;
const px = ref.subarray(16, 16 + H * W * 17);
const want = new Float32Array(ref.buffer.slice(ref.byteOffset + 16 + H * W * 17, ref.byteOffset + 16 + H * W * 17 + H * W * 16));
const tiles = new Uint8Array(5 * W * H * 4), plane = W * H * 4;
const chans = [[0, 1, 2], [3, 4, 5], [6, 6, 6], [7, 8, 9], [10, 11, 12]];
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
  const s = (y * W + x) * 17, d = ((H - 1 - y) * W + x) * 4;
  chans.forEach((cs, t) => { cs.forEach((c, j) => { tiles[t * plane + d + j] = px[s + c]; }); tiles[t * plane + d + 3] = 255; });
}

// infer.wasm's bytes (background skipping, as nts.js runs it)
const { instance } = await WebAssembly.instantiate(readFileSync(path.join(live, netDir, 'infer-relaxed.wasm')));
const k = instance.exports, put = b => { const p = k.nts_alloc(b.length); new Uint8Array(k.memory.buffer).set(b, p); return p; };
const net = readFileSync(path.join(live, netDir, 'net.bin'));
if (!k.nts_load(put(net), net.length)) throw new Error('net.bin did not load');
k.nts_set_fine(put(readFileSync(finePath)));
const inp = k.nts_alloc(W * H * 72); k.nts_prepare(put(tiles), W, H, Math.sin(pitch), Math.cos(pitch), inp);
const xa = k.nts_alloc(W * H * 256), xb = k.nts_alloc(W * H * 256), dist = k.nts_alloc(W * H), o = k.nts_alloc(W * H * 4);
k.nts_run(inp, W, H, 0, H, xa, xb, dist, o);
const wasm = new Uint8Array(k.memory.buffer, o, W * H * 4).slice();

// WebGPU in Chromium
const exe = ['/opt/pw-browsers/chromium-1194/chrome-linux/chrome'].find(existsSync);
const b = await chromium.launch({ executablePath: exe, args: ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--disable-vulkan-surface'] });
const page = await b.newPage();
page.on('console', m => { if (m.type() !== 'log') console.log(`[${m.type()}] ${m.text()}`); });
page.on('pageerror', e => console.log('[pageerror]', e.message));
const TYPES = { '.js': 'text/javascript', '.json': 'application/json', '.png': 'image/png', '.wasm': 'application/wasm' };
await page.route('https://nts.test/**', r => {
  const rel = new URL(r.request().url()).pathname.slice(1);
  if (!rel) return r.fulfill({ status: 200, body: '<!doctype html><title>gpu-nts check</title>', contentType: 'text/html' });
  const f = path.join(live, rel);
  if (!existsSync(f)) return r.fulfill({ status: 404, body: '' });
  r.fulfill({ status: 200, body: readFileSync(f), contentType: TYPES[path.extname(f)] || 'application/octet-stream' });
});
await page.goto('https://nts.test/');
const res = await page.evaluate(async ({ t64, W, H, pitch, netDir }) => {
  const { createGpuShader } = await import('/gpu-nts.js');
  const tiles = Uint8Array.from(atob(t64), c => c.charCodeAt(0));
  const s = await createGpuShader({ base: netDir, maxPixels: W * H, fallback: false });
  const ms = [];
  let rgba;
  for (let i = 0; i < 4; i++) { const t0 = performance.now(); rgba = await s.frame({ tiles, w: W, h: H, pitch }); ms.push(performance.now() - t0); }
  let bin = ''; for (let i = 0; i < rgba.length; i += 0x8000) bin += String.fromCharCode(...rgba.subarray(i, i + 0x8000));
  return { out: btoa(bin), ms, gpuMs: s.gpuMs, kernel: s.kernel, adapter: s.adapter };
}, { t64: Buffer.from(tiles).toString('base64'), W, H, pitch, netDir });
await b.close();
const gpu = Buffer.from(res.out, 'base64');

const over = (r, g, bl, a) => [r * a + 1 - a, g * a + 1 - a, bl * a + 1 - a];
function report(name, ref8, refF) {
  let worst = 0, sum = 0, off = 0, n = 0;
  for (let i = 0; i < W * H; i++) {
    const a = gpu[i * 4 + 3] / 255, g = over(gpu[i * 4] / 255, gpu[i * 4 + 1] / 255, gpu[i * 4 + 2] / 255, a);
    const r = refF ? over(refF[i * 4], refF[i * 4 + 1], refF[i * 4 + 2], refF[i * 4 + 3])
      : over(ref8[i * 4] / 255, ref8[i * 4 + 1] / 255, ref8[i * 4 + 2] / 255, ref8[i * 4 + 3] / 255);
    for (let c = 0; c < 3; c++) { const e = Math.abs(g[c] - r[c]); sum += e; n++; worst = Math.max(worst, e); if (e > 2 / 255) off++; }
  }
  console.log(`${name}: over white, largest difference ${worst.toExponential(2)}, mean ${(sum / n).toExponential(2)}, ${off} values off by > 2/255`);
}
console.log(`${W}x${H}, ${res.kernel} on ${res.adapter}; frame ${res.ms.map(m => m.toFixed(0)).join(' / ')} ms (first includes warm-up), GPU ${res.gpuMs?.toFixed(0) ?? 'n/a'} ms`);
report('WebGPU vs shader_nts.c', null, want);
{ const keep = Uint8Array.from(gpu); gpu.set(wasm); report('(infer.wasm bytes vs shader_nts.c, for scale)', null, want); gpu.set(keep); }
report('WebGPU vs infer.wasm ', wasm, null);
let bytesOff = 0; for (let i = 0; i < gpu.length; i++) if (Math.abs(gpu[i] - wasm[i]) > 1) bytesOff++;
console.log(`bytes differing from infer.wasm's by more than 1: ${bytesOff} of ${gpu.length}`);
