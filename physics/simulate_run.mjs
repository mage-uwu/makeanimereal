// Physics free run, offline: parcgen plans the auto-run lap on freerun.js's course, the trained
// tracker (web/tracker.js on MuJoCo's WASM build) makes the simulated body follow, and each new
// window is seeded where the body is (as the stage does). Prints progress and falls.
//
//   node physics/simulate_run.mjs <physics dir with humanoid.xml, tracker.bin/json> [seconds] [mujoco.js]
import fs from 'node:fs';
import { COURSE, rasterize } from '../web/freerun.js';
import { createTracker } from '../web/tracker.js';

const [dir, secondsArg, mjPath] = process.argv.slice(2);
const seconds = Number(secondsArg || 60);
const mujoco = await (await import(mjPath || new URL('../web/mujoco/mujoco.js', import.meta.url).href)).default();
const field = rasterize(COURSE);
const ground = (x, y) => field.heights[Math.max(0, Math.min(field.nx - 1, Math.round((x - field.minX) / field.cell))) * field.ny +
  Math.max(0, Math.min(field.ny - 1, Math.round((y - field.minY) / field.cell)))];
const tracker = await createTracker({ mujoco, xml: fs.readFileSync(`${dir}/humanoid.xml`, 'utf8'), weights: fs.readFileSync(`${dir}/tracker.bin`).buffer.slice(0),
  info: JSON.parse(fs.readFileSync(`${dir}/tracker.json`)), boxes: COURSE.boxes, ground });

// The generator (parcgen.wasm), synchronously.
const { instance } = await WebAssembly.instantiate(fs.readFileSync(new URL('../web/parcgen.wasm', import.meta.url)), {});
const g = instance.exports;
g.pg_create();
const w = fs.readFileSync(new URL('../web/parcgen.bin', import.meta.url));
new Uint8Array(g.memory.buffer, g.pg_weights(), w.length).set(w);
g.pg_bind();
new Float32Array(g.memory.buffer, g.pg_terrain(field.nx, field.ny, field.minX, field.minY, field.cell), field.nx * field.ny).set(field.heights);
const gauss = () => Math.sqrt(-2 * Math.log(Math.random() + 1e-12)) * Math.cos(2 * Math.PI * Math.random());
const generate = (prev, tx, ty) => {
  new Float32Array(g.memory.buffer, g.pg_prev(), 156).set(prev);
  const nz = new Float32Array(g.memory.buffer, g.pg_noise(), 15 * 91);
  for (let i = 0; i < nz.length; i++) nz[i] = gauss();
  g.pg_generate(tx, ty, 20, 0.65);
  return new Float32Array(g.memory.buffer, g.pg_out(), 15 * 78).slice();
};

const start = JSON.parse(fs.readFileSync(new URL('../web/parcgen-start.json', import.meta.url))).frames;
let frames = new Float32Array(15 * 78);
for (let f = 0; f < 15; f++) { frames.set(start[Math.min(f, 1)], f * 78); frames[f * 78] += COURSE.spawn[0]; frames[f * 78 + 1] += COURSE.spawn[1]; }
const at = i => frames.subarray(Math.min(14, i) * 78, Math.min(14, i) * 78 + 78);
tracker.reset(at(2), at(3));
let t = 2, lap = 1, falls = 0, time = 0, maxDist = 0, distSum = 0, steps = 0, laps = 0;
let body = tracker.frame();
while (time < seconds) {
  // Steer for the next corner from where the body is.
  let [gx, gy] = COURSE.lap[lap];
  if (Math.hypot(gx - body[0], gy - body[1]) < 2.5) { lap = (lap + 1) % COURSE.lap.length; if (lap === 1) laps++; [gx, gy] = COURSE.lap[lap]; }
  const d = Math.hypot(gx - body[0], gy - body[1]) || 1;
  // PARC's schedule: frames 2..5 of each window, the next window seeded from frames 4, 5, shifted
  // by the body's offset from the plan.
  if (t >= 5) {
    const prev = frames.slice(4 * 78, 6 * 78), now = at(t);
    for (const o of [0, 78]) { prev[o] += body[0] - now[0]; prev[o + 1] += body[1] - now[1]; }
    frames = generate(prev, (gx - body[0]) / d, (gy - body[1]) / d);
    t = 1;
  }
  const r = tracker.step(at(t + 1), at(t + 2));
  t += 1;
  time += 1 / 30; steps++;
  distSum += r.dist; maxDist = Math.max(maxDist, r.dist);
  if (r.fell) { falls++; tracker.reset(at(t), at(t + 1)); }
  body = tracker.frame();
  if (Math.abs(time - Math.round(time / 5) * 5) < 1 / 60) {
    console.log(`t ${time.toFixed(0).padStart(3)}s  body at (${body[0].toFixed(1)}, ${body[1].toFixed(1)}) z ${body[2].toFixed(2)} over ground ${ground(body[0], body[1]).toFixed(1)}  falls ${falls}  corner ${lap}`);
  }
}
console.log(`${seconds}s: ${laps} laps, ${falls} falls (${(falls / seconds * 60).toFixed(1)} per minute), mean tracking error ${(distSum / steps).toFixed(2)} m`);
