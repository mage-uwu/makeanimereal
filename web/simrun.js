// A simulated body on the stage: MoCapAct's multi-clip policy (npmp.js) drives dm_control's CMU
// humanoid in MuJoCo, tracking a CMU clip, and the character (box or VRM) is posed from the
// simulated body, not from the clip.
//
//   const run = await createSimRun({ THREE, vrm, mujoco, assets, clip, onStatus });
//   run.update(dt)     // advance the simulation (33 Hz) and pose the character in between
//   run.hips()         // where the character's hips are, for the camera
//
// Retargeting. At zero joint angles and an identity root, the CMU humanoid stands in VRM's own
// convention (y up, facing +z, left at +x), arms out in a T-pose, legs splayed 20 deg. So a VRM
// bone's world rotation is the matching body's rotation since that rest pose, MuJoCo's z-up world
// turned to y-up, times a fixed correction from the VRM T-pose direction of the bone to the CMU
// rest direction of the body (the legs' splay, the collarbones' lift). Local rotations follow from
// the VRM parents. The hips move with the CMU hip joints, scaled by leg length, with the feet's
// planted height matched.

// VRM bone, CMU body, the CMU body its direction points to, the VRM bone it points to.
const MAP = [
  ['hips', 'root'], ['spine', 'lowerback', 'upperback', 'chest'], ['chest', 'upperback', 'thorax', 'upperChest'],
  ['upperChest', 'thorax', 'lowerneck', 'neck'], ['neck', 'upperneck', 'head', 'head'], ['head', 'head'],
  ...['left', 'right'].flatMap(s => {
    const c = s[0];
    return [[`${s}Shoulder`, `${c}clavicle`, `${c}humerus`, `${s}UpperArm`], [`${s}UpperArm`, `${c}humerus`, `${c}radius`, `${s}LowerArm`],
      [`${s}LowerArm`, `${c}radius`, `${c}wrist`, `${s}Hand`], [`${s}Hand`, `${c}wrist`], [`${s}UpperLeg`, `${c}femur`, `${c}tibia`, `${s}LowerLeg`],
      [`${s}LowerLeg`, `${c}tibia`, `${c}foot`, `${s}Foot`], [`${s}Foot`, `${c}foot`, `${c}toes`, `${s}Toes`], [`${s}Toes`, `${c}toes`]];
  }),
];
const PARENT = { spine: 'hips', chest: 'spine', upperChest: 'chest', neck: 'upperChest', head: 'neck' };
for (const s of ['left', 'right']) Object.assign(PARENT, { [`${s}Shoulder`]: 'upperChest', [`${s}UpperArm`]: `${s}Shoulder`, [`${s}LowerArm`]: `${s}UpperArm`,
  [`${s}Hand`]: `${s}LowerArm`, [`${s}UpperLeg`]: 'hips', [`${s}LowerLeg`]: `${s}UpperLeg`, [`${s}Foot`]: `${s}LowerLeg`, [`${s}Toes`]: `${s}Foot` });
const FINGERS = ['Thumb', 'Index', 'Middle', 'Ring', 'Little'].flatMap(f => ['Metacarpal', 'Proximal', 'Intermediate', 'Distal'].map(s => f + s));
const PLANTED_FOOT = 0.016;                                   // CMU foot body height with the foot flat (clips' minimum)

export async function createSimRun({ THREE, vrm, mujoco, assets: { xml, weights, wasm, info }, clip, onStatus = () => {} }) {
  const { createNpmp } = await import('./npmp.js');
  const np = await createNpmp({ mujoco, xml, weights, wasm, info });
  const nq = info.nq, NB = info.bodies.length, names = info.body_names;
  const bodyId = n => (n === 'root' ? info.root : info.bodies[names.indexOf(n)]);
  const frame = t => {
    const f = clip.frames[Math.min(t, clip.frames.length - 1)];
    return { qpos: f.slice(0, nq), pos: f.slice(nq, nq + 3 * NB), quat: f.slice(nq + 3 * NB) };
  };

  // The CMU rest pose (identity root, zero joints) and the VRM rest pose, in the rig's own frame
  // (VRM 0.x rigs face -z there: F turns the +z-facing convention into it).
  const rest = new Float64Array(nq);
  rest[3] = 1;
  np.reset(rest, rest);
  const F = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), vrm.meta?.metaVersion === '0' ? Math.PI : 0);
  const R = (d, b, out = new THREE.Quaternion()) => out.set(d.xquat[4 * b + 1], d.xquat[4 * b + 2], d.xquat[4 * b + 3], d.xquat[4 * b]);
  const P = (d, b, out = new THREE.Vector3()) => out.set(d.xpos[3 * b], d.xpos[3 * b + 1], d.xpos[3 * b + 2]);
  vrm.humanoid.resetNormalizedPose();
  vrm.scene.updateMatrixWorld(true);
  const node = n => vrm.humanoid.getNormalizedBoneNode(n);
  const at = n => vrm.scene.worldToLocal(node(n).getWorldPosition(new THREE.Vector3()));
  const bones = MAP.filter(([v]) => node(v)).map(([v, c, cc, vc]) => {
    const restInv = R(np.data, bodyId(c)).invert();
    const fix = new THREE.Quaternion();
    if (cc && node(vc)) fix.setFromUnitVectors(at(vc).sub(at(v)).normalize(), P(np.data, bodyId(cc)).sub(P(np.data, bodyId(c))).applyQuaternion(F).normalize());
    let parent = PARENT[v];
    while (parent && !node(parent)) parent = PARENT[parent];
    return { v, b: bodyId(c), restInv, fix, parent };
  });
  const leg = n => at(`${n}UpperLeg`).distanceTo(at(`${n}LowerLeg`)) + at(`${n}LowerLeg`).distanceTo(at(`${n}Foot`));
  const cmuLeg = P(np.data, bodyId('lfemur')).distanceTo(P(np.data, bodyId('ltibia'))) + P(np.data, bodyId('ltibia')).distanceTo(P(np.data, bodyId('lfoot')));
  const k = (leg('left') + leg('right')) / 2 / cmuLeg;
  const legsMid = at('leftUpperLeg').add(at('rightUpperLeg')).multiplyScalar(0.5);
  const hipsAboveLegs = at('hips').sub(legsMid);
  const lift = at('leftFoot').y - k * PLANTED_FOOT;

  // Two snapshots of the simulated body, to draw in between: each mapped body's rotation, and the
  // midpoint of the hip joints.
  const snap = () => ({ q: bones.map(({ b }) => R(np.data, b)), legs: P(np.data, bodyId('lfemur')).add(P(np.data, bodyId('rfemur'))).multiplyScalar(0.5) });
  let t = 0, acc = 0, falls = 0, prev, cur;
  const start = from => {
    np.reset(frame(from).qpos, frame(from + 1).qpos);
    t = from;
    prev = cur = snap();
  };
  start(0);
  const last = clip.frames.length - 1 - Math.max(...info.ref_steps);
  const tick = () => {
    np.step(info.ref_steps.map(s => frame(t + s)));
    t++;
    prev = cur;
    if (np.error(frame(t)) > info.termination) { falls++; start(t); onStatus(`${falls} catch${falls === 1 ? '' : 'es'}: put back on the clip when it strays`); }
    else if (t >= last) start(0);
    else cur = snap();
  };

  // Z-up MuJoCo world to the VRM's y-up frame, (x, y, z) -> (x, z, -y), then into the rig's frame.
  const M = F.clone().multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2)), Finv = F.clone().invert();
  const world = new Map(), qa = new THREE.Quaternion(), inv = new THREE.Quaternion(), hipsWorld = new THREE.Vector3();
  function pose(w) {
    vrm.humanoid.resetNormalizedPose();
    bones.forEach((bn, i) => {
      qa.copy(prev.q[i]).slerp(cur.q[i], w);
      world.set(bn.v, M.clone().multiply(qa).multiply(bn.restInv).multiply(Finv).multiply(bn.fix));
    });
    for (const bn of bones) {
      const q = world.get(bn.v);
      node(bn.v).quaternion.copy(bn.parent ? inv.copy(world.get(bn.parent)).invert().multiply(q) : q);
    }
    for (const side of ['left', 'right']) for (const f of FINGERS) {
      const n = node(side + f);
      if (n) n.rotation.set(0, 0, (side === 'left' ? -1 : 1) * (f.startsWith('Thumb') ? 0.1 : f.endsWith('Proximal') ? 0.35 : 0.3));
    }
    const l = prev.legs.clone().lerp(cur.legs, w);
    const hips = node('hips');
    hips.position.set(l.x * k, l.z * k + lift, -l.y * k).applyQuaternion(F).add(hipsAboveLegs.clone().applyQuaternion(hips.quaternion));
    hipsWorld.copy(hips.position);
  }
  pose(0);

  return {
    update(dt) {
      acc += dt;
      for (let n = 0; acc >= info.control_dt && n < 4; n++) { acc -= info.control_dt; tick(); }
      acc = Math.min(acc, info.control_dt);
      pose(acc / info.control_dt);
    },
    hips: () => hipsWorld,
    falls: () => falls,
    dispose() { np.dispose(); vrm.humanoid.resetNormalizedPose(); },
  };
}
