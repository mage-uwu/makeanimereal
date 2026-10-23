// Where infer.wasm's time goes: one real frame (shader_nts --ref) through a -DPROF build, whose stages
// are separate functions, under V8's sampling profiler; self time per function.
//   clang ... -mrelaxed-simd -DPROF -o prof.wasm infer.c   (the build line in infer.c, plus -DPROF)
//   node profile_infer.mjs prof.wasm ref.bin fine.raw [seconds]
import { readFileSync } from 'node:fs';
import { Session } from 'node:inspector/promises';

const [wasmPath, refPath, finePath, secs = '6'] = process.argv.slice(2);
const k = (await WebAssembly.instantiate(readFileSync(wasmPath))).instance.exports;
const put = b => { const p = k.nts_alloc(b.length); new Uint8Array(k.memory.buffer).set(b, p); return p; };
// NTS_DIR: an export_nts.py output folder other than the viewer's
const net = readFileSync(process.env.NTS_DIR ? `${process.env.NTS_DIR}/net.bin` : new URL('../../live/nts/net.bin', import.meta.url));
if (!k.nts_load(put(net), net.length)) throw new Error('net.bin did not load');
k.nts_set_fine(put(readFileSync(finePath)));

const ref = readFileSync(refPath), dv = new DataView(ref.buffer, ref.byteOffset);
const H = dv.getInt32(4, true), W = dv.getInt32(8, true), pitch = dv.getFloat32(12, true) * Math.PI / 180;
const px = ref.subarray(16, 16 + H * W * 17), plane = W * H * 4, tiles = new Uint8Array(5 * plane);
const chans = [[0, 1, 2], [3, 4, 5], [6, 6, 6], [7, 8, 9], [10, 11, 12]];
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
  const s = (y * W + x) * 17, d = ((H - 1 - y) * W + x) * 4;
  chans.forEach((cs, t) => { cs.forEach((c, j) => { tiles[t * plane + d + j] = px[s + c]; }); tiles[t * plane + d + 3] = 255; });
}
const tp = put(tiles), inp = k.nts_alloc(W * H * 72);
const xa = k.nts_alloc(W * H * 256), xb = k.nts_alloc(W * H * 256), dist = k.nts_alloc(W * H), out = k.nts_alloc(W * H * 4);
const frame = () => { k.nts_prepare(tp, W, H, Math.sin(pitch), Math.cos(pitch), inp); k.nts_run(inp, W, H, 0, H, xa, xb, dist, out); };
for (let i = 0; i < 3; i++) frame();   // warm: optimised code

const s = new Session(); s.connect();
await s.post('Profiler.enable'); await s.post('Profiler.setSamplingInterval', { interval: 100 });
await s.post('Profiler.start');
let n = 0; const t0 = performance.now();
while (performance.now() - t0 < +secs * 1000) { frame(); n++; }
const ms = (performance.now() - t0) / n;
const { profile } = await s.post('Profiler.stop');

const self = new Map(), dt = new Map();
profile.samples.forEach((id, i) => dt.set(id, (dt.get(id) || 0) + (profile.timeDeltas[i] || 0)));
for (const node of profile.nodes) {
  const name = node.callFrame.functionName || '(anonymous)';
  self.set(name, (self.get(name) || 0) + (dt.get(node.id) || 0));
}
const total = [...self.values()].reduce((a, b) => a + b, 0);
console.log(`${W}x${H}, ${n} frames, ${ms.toFixed(1)} ms a frame on one thread`);
for (const [name, t] of [...self].sort((a, b) => b[1] - a[1]).slice(0, 14))
  if (t / total > 0.002) console.log(`${(100 * t / total).toFixed(1).padStart(5)}%  ${(ms * t / total).toFixed(1).padStart(6)} ms  ${name}`);
