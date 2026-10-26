// Run free run's auto-lap offline (no browser, no rendering): parcgen.wasm on freerun.js's course,
// PARC's exact schedule (every window seeded from frames 4, 5; 4 new frames each), steering for
// the lap's corners. Prints where she is each second: position, pelvis height above the ground
// under her, and whether her hands touch anything.
//
//   node parcgen/simulate.mjs [seconds] [cfg scale, -1 = off] [ddim stride] [weights]
import fs from 'node:fs';
import { COURSE, rasterize } from '../web/freerun.js';

const seconds = Number(process.argv[2] || 40);
const cfg = Number(process.argv[3] ?? -1), stride = Number(process.argv[4] || 10);
const weights = fs.readFileSync(process.argv[5] || new URL('../web/parcgen.bin', import.meta.url));
const { instance } = await WebAssembly.instantiate(fs.readFileSync(new URL('../web/parcgen.wasm', import.meta.url)), {});
const x = instance.exports;
x.pg_create();
new Uint8Array(x.memory.buffer, x.pg_weights(), weights.length).set(weights);
x.pg_bind();
const field = rasterize(COURSE);
new Float32Array(x.memory.buffer, x.pg_terrain(field.nx, field.ny, field.minX, field.minY, field.cell), field.nx * field.ny).set(field.heights);
const start = JSON.parse(fs.readFileSync(new URL('../web/parcgen-start.json', import.meta.url))).frames;
const ground = (px, py) => {
  const i = Math.max(0, Math.min(field.nx - 1, Math.round((px - field.minX) / field.cell)));
  const j = Math.max(0, Math.min(field.ny - 1, Math.round((py - field.minY) / field.cell)));
  return field.heights[i * field.ny + j];
};
const gauss = () => Math.sqrt(-2 * Math.log(Math.random() + 1e-12)) * Math.cos(2 * Math.PI * Math.random());
let frames = new Float32Array(15 * 78);
for (let f = 0; f < 15; f++) { frames.set(start[Math.min(f, 1)], f * 78); frames[f * 78] += COURSE.spawn[0]; frames[f * 78 + 1] += COURSE.spawn[1]; }
let lap = 1, laps = 0, t = 0, maxH = 0, msTot = 0, n = 0;
const prev = new Float32Array(x.memory.buffer, x.pg_prev(), 156), noise = new Float32Array(x.memory.buffer, x.pg_noise(), 15 * 91);
while (t < seconds) {
  const r = [frames[5 * 78], frames[5 * 78 + 1]];
  let [gx, gy] = COURSE.lap[lap];
  if (Math.hypot(gx - r[0], gy - r[1]) < 2.5) { lap = (lap + 1) % COURSE.lap.length; if (lap === 1) laps++; [gx, gy] = COURSE.lap[lap]; }
  const d = Math.hypot(gx - r[0], gy - r[1]);
  prev.set(frames.slice(4 * 78, 6 * 78));
  for (let i = 0; i < noise.length; i++) noise[i] = gauss();
  const t0 = performance.now();
  x.pg_generate((gx - r[0]) / d, (gy - r[1]) / d, stride, cfg);
  msTot += performance.now() - t0; n++;
  frames = new Float32Array(x.memory.buffer, x.pg_out(), 15 * 78).slice();
  for (let f = 2; f <= 5; f++) {
    t += 1 / 30;
    const p = frames.subarray(f * 78, f * 78 + 3), c = frames.subarray(f * 78 + 63, f * 78 + 78);
    const above = p[2] - ground(p[0], p[1]);
    maxH = Math.max(maxH, ground(p[0], p[1]));
    if (f === 2 && Math.abs(t - Math.round(t)) < 1 / 15) {
      console.log(`t ${t.toFixed(1).padStart(5)}s  at (${p[0].toFixed(1).padStart(5)}, ${p[1].toFixed(1).padStart(5)})  ground ${ground(p[0], p[1]).toFixed(1)} m  pelvis +${above.toFixed(2)} m  ` +
        `heading to corner ${lap}${c[5] > 0.5 || c[8] > 0.5 ? '  hands on' : ''}`);
    }
  }
}
console.log(`cfg ${cfg}, stride ${stride}, ${seconds}s: ${laps} full laps, highest ground stood on ${maxH.toFixed(1)} m, ${(msTot / n).toFixed(0)} ms per window (${n} windows)`);
