// Every exported clip through web/npmp.js (MuJoCo's WASM build), from its first frame, as
// mocapact/track_dm.py runs them in dm_control: steps tracked before the termination error passes
// 0.3, and the milliseconds a control step takes.
//   node mocapact/simulate.mjs [web/npmp]
import fs from 'node:fs';
import { createNpmp } from '../web/npmp.js';

const dir = process.argv[2] || 'web/npmp';
const mujoco = await (await import('../web/mujoco/mujoco.js')).default();
const info = JSON.parse(fs.readFileSync(`${dir}/npmp.json`));
const np = await createNpmp({ mujoco, xml: fs.readFileSync(`${dir}/cmu.xml`, 'utf8'), weights: fs.readFileSync(`${dir}/npmp.bin`).buffer.slice(0), wasm: fs.readFileSync(new URL('../web/npmp.wasm', import.meta.url)), info });
const nq = info.nq, B = info.bodies.length;
let ms = 0, n = 0;
for (const { id, label } of JSON.parse(fs.readFileSync(`${dir}/clips/index.json`))) {
  const frames = JSON.parse(fs.readFileSync(`${dir}/clips/${id}.json`)).frames;
  const frame = t => { const f = frames[Math.min(t, frames.length - 1)]; return { qpos: f.slice(0, nq), pos: f.slice(nq, nq + 3 * B), quat: f.slice(nq + 3 * B) }; };
  np.reset(frames[0].slice(0, nq), frames[1].slice(0, nq));
  const last = frames.length - 1 - 9;
  let t = 0, err = 0;
  while (t < last) {
    const t0 = performance.now();
    err = np.step(info.ref_steps.map(s => frame(t + s)));
    ms += performance.now() - t0; n++;
    t++;
    err = np.error(frame(t));
    if (err > info.termination) break;
  }
  console.log(`${id} ${label.padEnd(20)} ${t} of ${last} steps${t < last ? '  (strayed)' : ''}`);
}
console.log(`${(ms / n).toFixed(1)} ms per control step (${info.control_dt * 1000} ms of motion)`);
