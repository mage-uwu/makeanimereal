// infer.wasm against shader_nts.c on one frame (shader_nts --ref): the same inputs through the
// browser kernel, split into bands as the viewer splits them, then the largest and mean difference.
//   node check_infer.mjs ref.bin fine.raw [bands]
import { readFileSync } from 'node:fs';

const [refPath, finePath, bandsArg = '3'] = process.argv.slice(2);
const dir = new URL('../../live/nts/', import.meta.url);
const { instance } = await WebAssembly.instantiate(readFileSync(process.env.WASM || new URL('infer.wasm', dir)));
const k = instance.exports, mem = () => new Uint8Array(k.memory.buffer);
const put = bytes => { const p = k.nts_alloc(bytes.length); mem().set(bytes, p); return p; };

const net = readFileSync(new URL('net.bin', dir));
if (!k.nts_load(put(net), net.length)) throw new Error('net.bin did not load');
k.nts_set_fine(put(readFileSync(finePath)));

const ref = readFileSync(refPath);
const dv = new DataView(ref.buffer, ref.byteOffset);
const H = dv.getInt32(4, true), W = dv.getInt32(8, true), pitch = dv.getFloat32(12, true) * Math.PI / 180;
const px = ref.subarray(16, 16 + H * W * 17);
const want = new Float32Array(ref.buffer.slice(ref.byteOffset + 16 + H * W * 17, ref.byteOffset + 16 + H * W * 17 + H * W * 16));
// The five tiles as readPixels gives them: RGBA, rows bottom-up.
const tiles = new Uint8Array(5 * W * H * 4), plane = W * H * 4;
const chans = [[0, 1, 2], [3, 4, 5], [6, 6, 6], [7, 8, 9], [10, 11, 12]];
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
  const s = (y * W + x) * 17, d = ((H - 1 - y) * W + x) * 4;
  chans.forEach((cs, t) => { cs.forEach((c, j) => { tiles[t * plane + d + j] = px[s + c]; }); tiles[t * plane + d + 3] = 255; });
}
const tp = put(tiles), inp = k.nts_alloc(W * H * 18 * 4);
let t0 = performance.now();
k.nts_prepare(tp, W, H, Math.sin(pitch), Math.cos(pitch), inp);
const tPrep = performance.now() - t0;

// Bands as nts.js makes them: each band gets its rows plus the network's reach on both sides.
// sparse 0: every token (exactly the trainer's forward); 1: background skipped, as the viewer runs it.
const bands = +bandsArg, reach = k.nts_reach();
const xa = k.nts_alloc(W * H * 64 * 4), xb = k.nts_alloc(W * H * 64 * 4), dist = k.nts_alloc(W * H), outp = k.nts_alloc(W * H * 16);
console.log(`${W}x${H}, ${bands} bands, reach ${reach}, prepare ${tPrep.toFixed(0)} ms`);
for (const sparse of [0, 1]) for (let rep = 0; rep < 2; rep++) {   // the second time: optimised code
  const got = new Float32Array(H * W * 4);
  t0 = performance.now();
  for (let i = 0; i < bands; i++) {
    const r0 = Math.round(H * i / bands), r1 = Math.round(H * (i + 1) / bands);
    const a = Math.max(0, r0 - reach), b = Math.min(H, r1 + reach);
    k.nts_run_f(inp + a * W * 18 * 4, W, b - a, r0 - a, r1 - a, xa, xb, dist, sparse, outp);
    got.set(new Float32Array(k.memory.buffer, outp, (r1 - r0) * W * 4), r0 * W * 4);
  }
  const tRun = performance.now() - t0;
  if (!rep) continue;
  // as it's seen: composited over white (the held-out metric's way)
  let worst = 0, sum = 0, off = 0;
  for (let i = 0; i < got.length; i += 4) for (let c = 0; c < 3; c++) {
    const v = x => x[i + c] * x[i + 3] + 1 - x[i + 3], e = Math.abs(v(got) - v(want));
    sum += e; if (e > worst) worst = e; if (e > 2 / 255) off++;
  }
  console.log(`${sparse ? 'background skipped' : 'every token      '}: over white, largest difference ${worst.toExponential(2)}, mean ${(sum / got.length * 4 / 3).toExponential(2)}, ${off} values off by > 2/255; ${tRun.toFixed(0)} ms on one thread`);
}
