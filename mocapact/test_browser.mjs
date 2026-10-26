// web/npmp.js on MuJoCo's WASM build against the dm_control episode export_web.py --fixture recorded:
// the observations, intention and action at every recorded state, then a free rollout from the
// first state with the browser's own actions.
//   node mocapact/test_browser.mjs <fixture.json> [web/npmp]
import fs from 'node:fs';
import { createNpmp } from '../web/npmp.js';

const [fixturePath, dir = 'web/npmp'] = process.argv.slice(2);
const mujoco = await (await import('../web/mujoco/mujoco.js')).default();
const fx = JSON.parse(fs.readFileSync(fixturePath));
const info = JSON.parse(fs.readFileSync(`${dir}/npmp.json`));
const np = await createNpmp({ mujoco, xml: fs.readFileSync(`${dir}/cmu.xml`, 'utf8'), weights: fs.readFileSync(`${dir}/npmp.bin`).buffer.slice(0), wasm: fs.readFileSync(new URL('../web/npmp.wasm', import.meta.url)), info });
const clip = JSON.parse(fs.readFileSync(`${dir}/clips/${fx.clip}.json`));
const nq = info.nq, B = info.bodies.length;
const frame = t => { const f = clip.frames[Math.min(t, clip.frames.length - 1)]; return { qpos: f.slice(0, nq), pos: f.slice(nq, nq + 3 * B), quat: f.slice(nq + 3 * B) }; };
const refs = t => info.ref_steps.map(s => frame(t + s));
const maxDiff = (a, b) => Math.max(...Array.from(a, (v, i) => Math.abs(v - b[i])));

let enc = 0, dec = 0, zd = 0, ad = 0, worstEnc = -1;
fx.steps.forEach((s, k) => {
  np._setState(s.state, k ? fx.steps[k - 1].z : new Array(info.embed).fill(0));
  const r = np._act(refs(s.state.time_step));
  const e = maxDiff(r.enc, s.ref_encoder);
  if (e > enc) { enc = e; worstEnc = Array.from(r.enc, (v, i) => Math.abs(v - s.ref_encoder[i])).indexOf(e); }
  dec = Math.max(dec, maxDiff(r.dec, s.decoder)); zd = Math.max(zd, maxDiff(r.z, s.z)); ad = Math.max(ad, maxDiff(r.a, s.action));
});
console.log(`${fx.steps.length} states: max |JS - dm_control| encoder input ${enc.toExponential(1)} (at ${worstEnc}), decoder input ${dec.toExponential(1)}, z ${zd.toExponential(1)}, action ${ad.toExponential(1)}`);

np._setState(fx.steps[0].state, new Array(info.embed).fill(0));
const dq = [];
for (let k = 0; k + 1 < fx.steps.length; k++) {
  np.step(refs(fx.steps[k].state.time_step));
  dq.push(maxDiff(np.data.qpos, fx.steps[k + 1].state.qpos));
}
console.log(`free rollout: qpos difference after 1, 10, ${dq.length} steps: ${[0, 9, dq.length - 1].map(i => dq[i].toExponential(1)).join(', ')}`);
