// vrmanet in the browser: read a vrmanet .vrma (vrmanet/model.py's loader, in JS), load it into the
// WASM runtime (vrmanet/vrmanet.c, built by vrmanet/build-wasm.sh) and pose a three-vrm VRM with it.
//
//   const net = await createVrmanet(wasmBytes, vrmaBytes);
//   net.control({ walk: 1 }, { wave: 1 }); net.reset(); net.steady(); net.trigger('walk');
//   each frame: net.step(dt); net.apply(vrm);

const EXT = 'MAKEANIMEREAL_vrmanet';
const F = { W: 0, PERIOD: 1, A_LOG: 2, B: 3, C: 4, D: 5, REST_HIPS: 6, WIDTH: 7, U: 8, X: 9, Y: 10, GATES: 11, RATE: 12, POSE: 13 };
const I = { KEY_OFF: 0, KEY_CNT: 1, BONE_PART: 2 };
const COMPS = { SCALAR: 1, VEC3: 3, VEC4: 4 };

function readGlb(buf) {
  const dv = new DataView(buf), n = dv.getUint32(12, true);
  if (dv.getUint32(0, true) !== 0x46546c67) throw new Error('not a glTF binary');
  const js = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 20, n)));
  return { js, bin: buf.slice(28 + n, 28 + n + dv.getUint32(20 + n, true)) };
}

function accessor(js, bin, i) {
  const a = js.accessors[i], v = js.bufferViews[a.bufferView], c = COMPS[a.type];
  return { c, count: a.count, data: new Float32Array(bin, (v.byteOffset || 0) + (a.byteOffset || 0), a.count * c) };
}

export async function createVrmanet(wasmBytes, vrmaBytes) {
  const { js, bin } = readGlb(vrmaBytes);
  const ext = js.extensions[EXT], human = js.extensions.VRMC_vrm_animation.humanoid.humanBones;
  const byNode = Object.fromEntries(Object.entries(human).map(([k, v]) => [v.node, k]));
  const bones = Object.keys(byNode).map(Number).sort((a, b) => a - b).map(n => byNode[n]);
  const experts = ext.experts.map(x => x.name), parts = ext.parts;
  const E = experts.length, J = bones.length, P = parts.length, D = 4 * J + 3, hips = bones.indexOf('hips');
  const restHips = js.nodes[human.hips.node].translation || [0, 0, 0];

  // Decoder rows, one per keyframe (the loop's closing key repeats the first).
  const rows = [], keyOff = [], keyCnt = [], period = [];
  for (const x of ext.experts) {
    const anim = js.animations[x.animation];
    const times = anim.samplers.map(s => accessor(js, bin, s.input)).reduce((a, b) => (b.count > a.count ? b : a)).data;
    const K = times.length - 1;
    const W = new Float32Array(K * D);
    for (let k = 0; k < K; k++) {
      for (let j = 0; j < J; j++) W[k * D + 4 * j + 3] = 1;
      W.set(restHips, k * D + 4 * J);
    }
    for (const ch of anim.channels) {
      const s = anim.samplers[ch.sampler], v = accessor(js, bin, s.output), n = accessor(js, bin, s.input).count;
      const j = bones.indexOf(byNode[ch.target.node]), col = ch.target.path === 'rotation' ? 4 * j : 4 * J;
      if (ch.target.path !== 'rotation' && j !== hips) continue;
      for (let k = 0; k < K; k++) W.set(v.data.subarray((n === K + 1 ? k : 0) * v.c, (n === K + 1 ? k : 0) * v.c + v.c), k * D + col);
    }
    keyOff.push(rows.reduce((a, r) => a + r.length / D, 0));
    keyCnt.push(K);
    period.push(times[K]);
    rows.push(W);
  }
  const K = keyOff[E - 1] + keyCnt[E - 1];

  const { instance } = await WebAssembly.instantiate(wasmBytes, {});
  const x = instance.exports, s = ext.ssm;
  if (x.vn_create(E, J, P, s.states, s.inputs, s.outputs, K, hips) !== 0) throw new Error('model too big for the runtime');
  const f32 = (id, n) => new Float32Array(x.memory.buffer, x.vn_f32(id), n);
  const i32 = (id, n) => new Int32Array(x.memory.buffer, x.vn_i32(id), n);
  rows.forEach((W, e) => f32(F.W, K * D).set(W, keyOff[e] * D));
  f32(F.PERIOD, E).set(period);
  f32(F.A_LOG, s.states).set(accessor(js, bin, s.A_log).data);
  f32(F.B, s.states * s.inputs).set(accessor(js, bin, s.B).data);
  f32(F.C, s.outputs * s.states).set(accessor(js, bin, s.C).data);
  f32(F.D, s.outputs * s.inputs).set(accessor(js, bin, s.D).data);
  f32(F.REST_HIPS, 3).set(restHips);
  f32(F.WIDTH, 1).set([ext.kernelWidth]);
  i32(I.KEY_OFF, E).set(keyOff);
  i32(I.KEY_CNT, E).set(keyCnt);
  i32(I.BONE_PART, J).set(bones.map(b => ext.boneParts[b]));
  const u = f32(F.U, s.inputs), pose = f32(F.POSE, D), gates = f32(F.GATES, E * P);

  return {
    experts, parts, bones, credit: js.asset.copyright || '',
    // body = { expert: weight } on every part; arms = the same for the arms only; rate = { expert: rate }.
    control(body, arms = null, rate = {}) {
      u.fill(0);
      for (let p = 0; p < P; p++) for (const [n, w] of Object.entries((arms && parts[p] === 'arms' ? arms : body) || {})) {
        if (experts.includes(n)) u[experts.indexOf(n) * P + p] = w;
      }
      for (const [n, r] of Object.entries(rate)) if (experts.includes(n)) u[E * P + experts.indexOf(n)] = r - 1;
    },
    reset: () => x.vn_reset(),
    steady: () => x.vn_steady(),
    trigger: name => experts.includes(name) && x.vn_trigger(experts.indexOf(name)),
    step: dt => x.vn_step(dt),
    gate: (name, part) => gates[experts.indexOf(name) * P + parts.indexOf(part)],
    // Write the pose onto the VRM's normalized bones; the hips' path scales with the hips' height.
    apply(vrm) {
      for (let j = 0; j < J; j++) vrm.humanoid.getNormalizedBoneNode(bones[j])?.quaternion.fromArray(pose, 4 * j);
      const node = vrm.humanoid.getNormalizedBoneNode('hips'), scale = vrm.humanoid.normalizedRestPose.hips.position[1] / restHips[1];
      node.position.set(pose[4 * J] * scale, pose[4 * J + 1] * scale, pose[4 * J + 2] * scale);
    },
  };
}
