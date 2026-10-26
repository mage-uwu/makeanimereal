// MoCapAct's multi-clip policy in the browser: dm_control's CMU humanoid simulated by MuJoCo's
// WASM build, driven by the NPMP (a recurrent reference encoder to a 60-d motor intention, and a
// decoder from intention and body state to the 56 position actuators), following reference motion.
// Everything here mirrors dm_control's MultiClipMocapTracking observations (mocapact/export_web.py
// records where each lives) and MoCapAct's NpmpPolicy, checked step for step against them by
// mocapact/test_browser.mjs.
//
//   const np = await createNpmp({ mujoco, xml, weights, wasm, info });   // npmp.bin, npmp.wasm, npmp.json
//   np.reset(qpos, nextQpos)      // start in a reference pose, moving toward the next frame
//   np.step(refs)                 // one control step (0.03 s); refs: the next 5 reference frames,
//                                 // each { pos: 31 x 3, quat: 31 x 4 (w, x, y, z) } of the tracked bodies
//   np.reference(qpos)            // a reference frame from a pose (forward kinematics)
//   np.data                       // MuJoCo's state: xpos, xquat of every body, for rendering

export async function createNpmp({ mujoco, xml, weights, wasm, info }) {
  const model = mujoco.MjModel.from_xml_string(xml);
  const data = new mujoco.MjData(model);
  const ghost = new mujoco.MjData(model);                                 // for reference poses
  const B = info.bodies.length;

  // The network: npmp.bin (float16, in export_web.py's layout order) as float32 in npmp.wasm's
  // memory, which runs the layers (SIMD); the input normalizers stay here.
  const { instance } = await WebAssembly.instantiate(wasm, {});
  const k = instance.exports;
  const f16 = new Uint16Array(weights), W = {};
  const half = h => { const e = (h >> 10) & 31, m = h & 1023, s = h >> 15 ? -1 : 1;
    return e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15); };
  const table = new Float32Array(65536).map((_, h) => half(h));
  let at = 0;
  for (const [name, shape] of info.layout) {
    const n = shape.reduce((a, b) => a * b, 1), ptr = k.np_alloc(n);
    new Float32Array(k.memory.buffer, ptr, n).set(f16.subarray(at, at + n).reduce((o, h, i) => (o[i] = table[h], o), new Float32Array(n)));
    W[name] = { ptr, n, values: name.endsWith('_mean') || name.endsWith('_std') ? new Float32Array(k.memory.buffer, ptr, n).slice() : null };
    at += n;
  }
  const H = 1024, E = info.embed;
  const buf = n => ({ ptr: k.np_alloc(n), n });
  const io = { enc: buf(W['encoder.0.W'].n / H), dec: buf(W['decoder.0.W'].n / H), h: [buf(H), buf(H), buf(H)], z: buf(2 * E), a: buf(info.nu) };
  const view = b => new Float32Array(k.memory.buffer, b.ptr, b.n);
  // Linear, LayerNorm and ELU per hidden layer, then a linear output (sizes: each layer's width).
  const mlp = (net, idx, x, sizes, out) => {
    let src = x;
    idx.forEach((i, l) => {
      const dst = l === idx.length - 1 ? out : io.h[l];
      k.np_linear(dst.ptr, src.ptr, W[`${net}.${i}.W`].ptr, W[`${net}.${i}.b`].ptr, src.n, sizes[l]);
      if (l < idx.length - 1) k.np_norm_elu(dst.ptr, W[`${net}.${i + 1}.g`].ptr, W[`${net}.${i + 1}.beta`].ptr, sizes[l]);
      src = dst;
    });
  };

  // dm_control's observables, from MuJoCo's arrays.
  const proprio = () => {
    const d = data, r = info.root, o = [];
    for (const a of info.joints_qpos) o.push(d.qpos[a]);
    for (const a of info.joints_dof) o.push(d.qvel[a]);
    for (const a of info.sensors_velocimeter) o.push(d.sensordata[a]);
    for (const a of info.sensors_gyro) o.push(d.sensordata[a]);
    for (const a of info.end_effectors_pos) o.push(d.sensordata[a]);
    o.push(d.xmat[9 * r + 6], d.xmat[9 * r + 7], d.xmat[9 * r + 8]);              // world_zaxis
    for (let i = 0; i < info.nu; i++) o.push(d.act[i]);                          // actuator_activation
    for (const a of info.sensors_touch) o.push(d.sensordata[a] > info.touch_threshold ? 1 : 0);
    for (const a of info.sensors_torque) o.push(Math.tanh(2 * d.sensordata[a] / info.torque_threshold));
    return o;
  };
  const reference = refs => {
    const d = data, r = info.root, X = d.xmat, pos = [], quat = [];
    for (const ref of refs) for (let k = 0; k < B; k++) {                       // reference_rel_bodies_pos_local
      const b = info.bodies[k], v = [ref.pos[3 * k] - d.xpos[3 * b], ref.pos[3 * k + 1] - d.xpos[3 * b + 1], ref.pos[3 * k + 2] - d.xpos[3 * b + 2]];
      for (let j = 0; j < 3; j++) pos.push(v[0] * X[9 * r + j] + v[1] * X[9 * r + 3 + j] + v[2] * X[9 * r + 6 + j]);
    }
    for (const ref of refs) for (let k = 0; k < B; k++) {                       // reference_rel_bodies_quats: conj(body) * ref
      const b = info.bodies[k], [aw, ax, ay, az] = [d.xquat[4 * b], -d.xquat[4 * b + 1], -d.xquat[4 * b + 2], -d.xquat[4 * b + 3]];
      const [bw, bx, by, bz] = [ref.quat[4 * k], ref.quat[4 * k + 1], ref.quat[4 * k + 2], ref.quat[4 * k + 3]];
      quat.push(aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw);
    }
    return [d.xpos[3 * r + 2], ...pos, ...quat];                               // body_height first
  };
  const normalize = (x, net, into) => {
    const m = W[`${net}_mean`].values, sd = W[`${net}_std`].values, v = view(into);
    for (let i = 0; i < x.length; i++) v[i] = (x[i] - m[i]) / sd[i];
    return v.subarray(0, x.length);
  };

  let z = new Float32Array(E);
  // One policy evaluation: features, the next intention z, the action.
  function act(refs) {
    const p = proprio();
    const enc = normalize([...p, ...reference(refs)], 'ref_encoder', io.enc).slice();
    view(io.enc).set(z, enc.length);
    mlp('encoder', [0, 3, 6], io.enc, [H, H, 2 * E], io.z);
    z = view(io.z).slice(0, E);
    const dec = normalize(p, 'decoder', io.dec).slice();
    view(io.dec).set(z, dec.length);
    mlp('decoder', [0, 3, 6, 9], io.dec, [H, H, H, info.nu], io.a);
    const a = view(io.a).map(v => Math.max(-1, Math.min(1, v)));
    return { a, enc, dec, z };
  }

  return {
    model, data,
    // Put the body in pose q (qpos), moving toward q2 over one control step.
    reset(q, q2) {
      mujoco.mj_resetData(model, data);
      data.qpos.set(q);
      const dt = info.control_dt, v = new Float64Array(info.nv);
      for (let i = 0; i < 3; i++) v[i] = (q2[i] - q[i]) / dt;
      // Root angular velocity, in the root's frame: conj(q) * q2 as an axis-angle, per second.
      const [aw, ax, ay, az] = [q[3], -q[4], -q[5], -q[6]], [bw, bx, by, bz] = [q2[3], q2[4], q2[5], q2[6]];
      let d = [aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw];
      if (d[0] < 0) d = d.map(x => -x);
      const sn = Math.hypot(d[1], d[2], d[3]), ang = 2 * Math.atan2(sn, d[0]) / dt;
      for (let i = 0; i < 3; i++) v[3 + i] = sn > 1e-9 ? d[1 + i] / sn * ang : 0;
      for (let i = 7; i < info.nq; i++) v[i - 1] = (q2[i] - q[i]) / dt;
      data.qvel.set(v);
      mujoco.mj_forward(model, data);
      z = new Float32Array(E);
    },
    // One control step toward refs[0]: the action, then 6 physics steps as dm_control takes them
    // (mj_step2 then mj_step1), and a forward pass so every sensor, contact and torque included,
    // reads the new state as dm_control's observations do.
    step(refs) {
      const { a } = act(refs);
      data.ctrl.set(a);
      for (let i = 0; i < info.substeps; i++) { mujoco.mj_step2(model, data); mujoco.mj_step1(model, data); }
      mujoco.mj_forward(model, data);
      return this.error(refs[0]);
    },
    // dm_control's termination error against a reference frame (body positions; joints if given).
    error(ref) {
      let eb = 0;
      for (let k = 0; k < B; k++) for (let j = 0; j < 3; j++) eb += Math.abs(ref.pos[3 * k + j] - data.xpos[3 * info.bodies[k] + j]);
      eb /= 3 * B;
      if (!ref.qpos) return eb;
      let ej = 0;
      for (const a of info.mocap_joints_qpos) ej += Math.abs(ref.qpos[a] - data.qpos[a]);
      return 0.5 * eb + 0.5 * ej / info.mocap_joints_qpos.length;
    },
    // A reference frame from a pose: the tracked bodies' positions and quaternions (w, x, y, z).
    reference(q) {
      ghost.qpos.set(q);
      mujoco.mj_kinematics(model, ghost);
      const pos = new Float64Array(3 * B), quat = new Float64Array(4 * B);
      info.bodies.forEach((b, k) => { pos.set(ghost.xpos.subarray(3 * b, 3 * b + 3), 3 * k); quat.set(ghost.xquat.subarray(4 * b, 4 * b + 4), 4 * k); });
      return { pos, quat, qpos: q };
    },
    dispose() { ghost.delete(); data.delete(); model.delete(); },
    // For tests (mocapact/test_browser.mjs).
    _act: act,
    _setState({ qpos, qvel, act: activation, ctrl, sensordata, qacc_warmstart: warm, time }, zPrev) {
      data.qpos.set(qpos); data.qvel.set(qvel); data.act.set(activation); data.ctrl.set(ctrl); data.time = time;
      mujoco.mj_forward(model, data);
      if (warm) data.qacc_warmstart.set(warm);                                  // the solver's starting point
      data.sensordata.set(sensordata);                                           // as the last step left them
      if (zPrev) z = Float32Array.from(zPrev);
    },
  };
}
