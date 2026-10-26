// Replay physics/test_browser.py's rollout with web/tracker.js on MuJoCo's WASM build and compare.
//   node physics/test_browser.mjs <out_dir> <mujoco.js>
import fs from 'node:fs';
import { createTracker } from '../web/tracker.js';
const [dir, mjPath] = process.argv.slice(2);
const mujoco = await (await import(mjPath)).default();
const r = JSON.parse(fs.readFileSync(`${dir}/rollout.json`));
const ground = (x, y) => {
  const i = Math.max(0, Math.min(r.hf.length - 1, Math.round((x - r.min_point[0]) / r.dx)));
  const j = Math.max(0, Math.min(r.hf[0].length - 1, Math.round((y - r.min_point[1]) / r.dx)));
  return r.hf[i][j];
};
const tr = await createTracker({ mujoco, xml: fs.readFileSync(`${dir}/humanoid.xml`, 'utf8'), weights: fs.readFileSync(`${dir}/tracker.bin`).buffer.slice(0),
  info: JSON.parse(fs.readFileSync(`${dir}/tracker.json`)), boxes: r.boxes, ground });
const F = r.frames.map(f => Float32Array.from(f));
tr.reset(F[r.t0], F[r.t0 + 1]);
let worstObs = 0, worstQ = [];
for (let k = 0; k < r.qpos.length; k++) {
  if (k === 0) { const o = tr._observe(F[r.t0 + 1], F[r.t0 + 2]); worstObs = Math.max(...o.map((v, i) => Math.abs(v - r.obs0[i]))); }
  tr.step(F[r.t0 + k + 1], F[r.t0 + k + 2]);
  const q = tr._qpos();
  worstQ.push(Math.max(...r.qpos[k].map((v, i) => Math.abs(v - q[i]))));
}
console.log(`first observation: max |JS - Python| ${worstObs.toExponential(2)}`);
console.log(`qpos difference after 1, 10, 30, ${worstQ.length} steps: ${[0, 9, 29, worstQ.length - 1].map(i => (worstQ[i] ?? NaN).toExponential(1)).join(', ')}`);
