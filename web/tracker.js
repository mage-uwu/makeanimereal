// The physics tracker in the browser: PARC's humanoid simulated by MuJoCo (the official WASM build,
// the same engine version it was trained in) and driven by the PPO policy trained in physics/
// (tracker.bin), following reference frames. A frame is PARC's 78 floats: root position 3, root
// rotation 4 (x, y, z, w), 14 joint rotations 56, contacts 15. Everything here mirrors
// physics/env.py (observations) and physics/body.py (hinges) line for line.
//
//   const tr = await createTracker({ mujoco, xml, weights, info, boxes, ground });
//   tr.reset(frameA, frameB);        // start in the reference pose (frameA), moving toward frameB
//   const { dist } = tr.step(ref1, ref2);   // one 30 Hz step toward ref1 (looking ahead to ref2)
//   tr.frame()                        // the simulated body as a PARC frame, for rendering

const BODIES = 15, PARENT = [-1, 0, 1, 1, 3, 4, 1, 6, 7, 0, 9, 10, 0, 12, 13];
const OFFSET = [[0, 0, 0], [0, 0, .236151], [0, 0, .223894], [-.02405, -.18311, .2435], [0, -.274788, 0], [0, -.258947, 0],
  [-.02405, .18311, .2435], [0, .274788, 0], [0, .258947, 0], [0, -.084887, 0], [0, 0, -.421546], [0, 0, -.40987],
  [0, .084887, 0], [0, 0, -.421546], [0, 0, -.40987]];
// Hinges per body (after the root), in qpos order: 'yxz' / 'xyz' ball joints, or a hinge axis.
const JOINTS = [['yxz'], ['xyz'], ['xyz'], [[0, 0, 1]], null, ['xyz'], [[0, 0, 1]], null, ['yxz'], [[0, 1, 0]], ['xyz'], ['yxz'], [[0, 1, 0]], ['xyz']];
const END_EFFECTORS = [5, 8, 11, 14];
const GRID = [];
for (let i = 0; i < 9; i++) for (let j = 0; j < 9; j++) GRID.push([-1.2 + i * 0.4, -1.6 + j * 0.4]);

// --- quaternions (x, y, z, w) ---------------------------------------------------------------------
const qmul = (a, b) => [a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1], a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
  a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3], a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]];
const qconj = q => [-q[0], -q[1], -q[2], q[3]];
const qrot = (q, v) => {
  const t = [2 * (q[1] * v[2] - q[2] * v[1]), 2 * (q[2] * v[0] - q[0] * v[2]), 2 * (q[0] * v[1] - q[1] * v[0])];
  return [v[0] + q[3] * t[0] + q[1] * t[2] - q[2] * t[1], v[1] + q[3] * t[1] + q[2] * t[0] - q[0] * t[2], v[2] + q[3] * t[2] + q[0] * t[1] - q[1] * t[0]];
};
const heading = q => { const d = qrot(q, [1, 0, 0]); return Math.atan2(d[1], d[0]); };
const zrot = a => [0, 0, Math.sin(a / 2), Math.cos(a / 2)];
const axisQuat = (axis, a) => [axis[0] * Math.sin(a / 2), axis[1] * Math.sin(a / 2), axis[2] * Math.sin(a / 2), Math.cos(a / 2)];
const AX = { x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1] };
const wrap = x => ((x + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;

// Reference frame -> body positions and rotations (PARC's forward kinematics).
function fk(f) {
  const pos = [[f[0], f[1], f[2]]], rot = [[f[3], f[4], f[5], f[6]]];
  for (let k = 1; k < BODIES; k++) {
    const p = PARENT[k], o = qrot(rot[p], OFFSET[k]);
    pos.push([pos[p][0] + o[0], pos[p][1] + o[1], pos[p][2] + o[2]]);
    rot.push(qmul(rot[p], [f[7 + 4 * (k - 1)], f[8 + 4 * (k - 1)], f[9 + 4 * (k - 1)], f[10 + 4 * (k - 1)]]));
  }
  return { pos, rot };
}

// Reference frame -> the 28 hinge angles (physics/body.py: frames_to_qpos), continuous with prev.
function hinges(f, prev) {
  const out = [];
  for (let k = 0; k < 14; k++) {
    const J = JOINTS[k];
    if (!J) continue;
    const q = [f[7 + 4 * k], f[8 + 4 * k], f[9 + 4 * k], f[10 + 4 * k]];
    if (typeof J[0] !== 'string') {                                     // one hinge: the angle about its axis
      const s = q[0] * J[0][0] + q[1] * J[0][1] + q[2] * J[0][2];
      out.push(2 * Math.atan2(s, q[3]));
      continue;
    }
    const [x, y, z, w] = q;
    const R = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]];
    let one;
    if (J[0] === 'xyz') one = [Math.atan2(-R[1][2], R[2][2]), Math.asin(Math.max(-1, Math.min(1, R[0][2]))), Math.atan2(-R[0][1], R[0][0])];
    else one = [Math.atan2(R[0][2], R[2][2]), Math.asin(Math.max(-1, Math.min(1, -R[1][2]))), Math.atan2(R[1][0], R[1][1])];
    const alt = [wrap(one[0] + Math.PI), wrap(Math.PI - one[1]), wrap(one[2] + Math.PI)];
    const base = out.length;
    if (!prev) out.push(...(Math.abs(one[1]) <= Math.abs(alt[1]) ? one : alt));
    else {
      const near = c => c.map((v, i) => prev[base + i] + wrap(v - prev[base + i]));
      const a = near(one), b = near(alt);
      const da = a.reduce((s, v, i) => s + Math.abs(v - prev[base + i]), 0), db = b.reduce((s, v, i) => s + Math.abs(v - prev[base + i]), 0);
      out.push(...(da <= db ? a : b));
    }
  }
  return out;
}

export async function createTracker({ mujoco, xml, weights, info, boxes, ground }) {
  // The model: the humanoid plus this course's floor and boxes.
  const geoms = ['<geom name="floor" type="plane" size="100 100 .2" condim="3" friction="1 .05 .05"/>',
    ...boxes.map(([x0, x1, y0, y1, h], i) => `<geom name="box${i}" type="box" condim="3" friction="1 .05 .05" ` +
      `pos="${(x0 + x1) / 2} ${(y0 + y1) / 2} ${h / 2}" size="${(x1 - x0) / 2} ${(y1 - y0) / 2} ${h / 2}"/>`)].join('');
  const model = mujoco.MjModel.from_xml_string(xml.replace('</worldbody>', geoms + '</worldbody>'));
  const data = new mujoco.MjData(model);

  // The policy: observation normalization, then ReLU layers.
  const w = new Float32Array(weights), O = info.obs;
  let at = 0;
  const take = n => w.subarray(at, (at += n));
  const mean = take(O), std = take(O);
  const layers = info.layers.map(([i, o]) => ({ i, o, W: take(i * o), b: take(o) }));
  const policy = obs => {
    let x = Float32Array.from(obs, (v, k) => Math.max(-5, Math.min(5, (v - mean[k]) / std[k])));
    layers.forEach(({ i, o, W, b }, l) => {
      const y = new Float32Array(o);
      for (let c = 0; c < o; c++) y[c] = b[c];
      for (let r = 0; r < i; r++) { const xr = x[r]; if (xr === 0) continue; const row = r * o; for (let c = 0; c < o; c++) y[c] += xr * W[row + c]; }
      if (l < layers.length - 1) for (let c = 0; c < o; c++) y[c] = y[c] > 0 ? y[c] : 0;
      x = y;
    });
    return x;
  };

  // The simulated body: positions and rotations (x, y, z, w) of the 15 bodies (ids 1..15).
  const state = () => {
    const pos = [], rot = [];
    for (let k = 0; k < BODIES; k++) {
      pos.push([data.xpos[3 * (k + 1)], data.xpos[3 * (k + 1) + 1], data.xpos[3 * (k + 1) + 2]]);
      const o = 4 * (k + 1);
      rot.push([data.xquat[o + 1], data.xquat[o + 2], data.xquat[o + 3], data.xquat[o]]);
    }
    return { pos, rot };
  };

  // physics/env.py: observe().
  function observe(ref1, ref2) {
    const { pos, rot } = state(), root = pos[0], rq = rot[0], h = heading(rq), hinv = zrot(-h);
    const c = Math.cos(h), s = Math.sin(h);
    const heights = GRID.map(([gx, gy]) => ground(gx * c - gy * s + root[0], gx * s + gy * c + root[1]));
    const tn = q => [...qrot(q, [1, 0, 0]), ...qrot(q, [0, 0, 1])];
    const v = data.qvel;
    const obs = [root[2] - heights[40]];
    for (let k = 1; k < BODIES; k++) obs.push(...qrot(hinv, [pos[k][0] - root[0], pos[k][1] - root[1], pos[k][2] - root[2]]));
    for (let k = 0; k < BODIES; k++) obs.push(...tn(qmul(hinv, rot[k])));
    obs.push(...qrot(hinv, [v[0], v[1], v[2]]), ...qrot(hinv, qrot(rq, [v[3], v[4], v[5]])));
    for (let i = 6; i < 34; i++) obs.push(v[i] * 0.1);
    const refs = [fk(ref1), fk(ref2)];
    for (const r of refs) for (let k = 0; k < BODIES; k++) obs.push(...qrot(hinv, [r.pos[k][0] - root[0], r.pos[k][1] - root[1], r.pos[k][2] - root[2]]));
    for (const r of refs) for (let k = 0; k < BODIES; k++) obs.push(...tn(qmul(hinv, r.rot[k])));
    for (const g of heights) obs.push(g - root[2]);
    return { obs, pos };
  }

  let target = null;                                                     // the last hinge targets (for continuity)
  return {
    // Put the body in frame a, moving as it moves from a to b over one frame (1/30 s).
    reset(a, b) {
      mujoco.mj_resetData(model, data);
      const ha = hinges(a), hb = hinges(b, ha);
      data.qpos.set([a[0], a[1], a[2], a[6], a[3], a[4], a[5], ...ha]);
      const dq = qmul(qconj([a[3], a[4], a[5], a[6]]), [b[3], b[4], b[5], b[6]]), sgn = dq[3] < 0 ? -1 : 1;
      const sn = Math.hypot(dq[0], dq[1], dq[2]), ang = 2 * Math.atan2(sn, sgn * dq[3]) * 30;
      const axis = sn > 1e-9 ? [sgn * dq[0] / sn, sgn * dq[1] / sn, sgn * dq[2] / sn] : [0, 0, 0];
      data.qvel.set([(b[0] - a[0]) * 30, (b[1] - a[1]) * 30, (b[2] - a[2]) * 30, axis[0] * ang, axis[1] * ang, axis[2] * ang,
        ...hb.map((v, i) => (v - ha[i]) * 30)]);
      mujoco.mj_forward(model, data);
      target = ha;
    },
    // One 30 Hz step toward ref1: the policy's offsets on the reference hinges, then 10 physics steps.
    step(ref1, ref2) {
      const { obs } = observe(ref1, ref2);
      const act = policy(obs);
      target = hinges(ref1, target);
      for (let i = 0; i < 28; i++) data.ctrl[i] = target[i] + act[i];
      for (let i = 0; i < info.substeps; i++) mujoco.mj_step(model, data);
      const pos = state().pos, ref = fk(ref1).pos;
      let dist = 0;
      for (let k = 0; k < BODIES; k++) dist = Math.max(dist, Math.hypot(pos[k][0] - ref[k][0], pos[k][1] - ref[k][1], pos[k][2] - ref[k][2]));
      return { dist, fell: dist > info.terminate || !Number.isFinite(dist) };
    },
    // The simulated body as a PARC frame (physics/body.py: qpos_to_frames).
    frame() {
      const p = data.qpos, f = new Float32Array(78);
      f.set([p[0], p[1], p[2], p[4], p[5], p[6], p[3]]);
      let col = 7;
      for (let k = 0; k < 14; k++) {
        const J = JOINTS[k];
        let q = [0, 0, 0, 1];
        if (J && typeof J[0] === 'string') {
          const o = J[0];
          q = qmul(qmul(axisQuat(AX[o[0]], p[col]), axisQuat(AX[o[1]], p[col + 1])), axisQuat(AX[o[2]], p[col + 2]));
          col += 3;
        } else if (J) { q = axisQuat(J[0], p[col]); col += 1; }
        f.set(q, 7 + 4 * k);
      }
      return f;
    },
    dispose() { data.delete(); model.delete(); },
    // For tests (physics/test_browser.mjs).
    _observe: (ref1, ref2) => observe(ref1, ref2).obs,
    _qpos: () => Array.from(data.qpos),
  };
}
