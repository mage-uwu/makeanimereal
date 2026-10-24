// A two-handed longsword: built from a few lathed and extruded pieces (no model file), held on the
// normalized right hand bone, whose axes are the same on any VRM: in the T-pose the fingers point
// along -X (the left hand's along +X), the palms face -Y and her front is +Z.
//
//   const sword = createSword({ THREE, vrm });   // after the model loads; hidden until shown
//   sword.show(true);                            // she draws it
//   sword.grip();                                // each frame, after the pose: hands onto the hilt
//   sword.ends(a, b);                            // its pommel and tip, world space (for the crop box)
//
// The hands are fitted to the hilt, not the hilt to the mocap. The right fist is measured from her
// own finger bones (between the index and little knuckles, a little palm-ward), the grip runs through
// it from the little finger to the index, and the fingers close round it. With both hands on it (the
// mocap's fists close together, the left one below the right along the grip: both slash clips are
// two-handed), the hilt is the line through the two fists: the right wrist turns so the sword,
// seated in that fist, points from the left fist through the right, so its angle always agrees with
// both hands, whatever the mocap's right wrist did (the actor's real sword forced the wrist the same way). The left hand is then pulled onto that line a fist below the right:
// turned so its fist lies along the grip, and the arm solved to it (two-bone IK keeping the elbow's
// plane). One-handed, the sword follows the right hand alone and the left arm the mocap, with smooth
// hand-offs between.
//
// The network sees the sword like her other meshes: colour from its own flat materials, normals,
// depth and UVs from the geometry, and `part` (userData.partPick: the first of the character's part
// names to match) picks which part's neural texture it samples: her front hair's for the steel, the
// shorts' (a colour-only part, so the network follows the colour) for the gold, her dress's for the grip.
const PART = { steel: [/Hair_00_HAIR/], gold: [/^Shorts$/, /Tops_01_CLOTH/], navy: [/Onepiece_00_CLOTH_01/, /CLOTH/] };
const pick = res => names => { for (const re of res) { const i = names.findIndex(n => re.test(n)); if (i >= 0) return i; } return 0; };
const FINGERS = ['Index', 'Middle', 'Ring', 'Little'];
// two hands on the hilt: the mocap's fists within NEAR metres of each other fully, apart past FAR
const NEAR = 0.14, FAR = 0.3;

export function createSword({ THREE, vrm }) {
  const bone = n => vrm.humanoid.getNormalizedBoneNode(n);
  const V = () => new THREE.Vector3(), Q = () => new THREE.Quaternion();
  const root = new THREE.Group();
  root.name = 'sword';
  root.visible = false;
  const mat = (color, name, part) => Object.assign(new THREE.MeshBasicMaterial({ color, name }), { userData: { part } });
  const steel = mat(0xcfd6e2, 'SwordBlade', PART.steel), edge = mat(0xf2f5fa, 'SwordEdge', PART.steel), gold = mat(0xc9a24a, 'SwordGuard', PART.gold), wrap = mat(0x1c2056, 'SwordGrip', PART.navy);

  // lengths in metres along the grip (+Z, toward the blade), from the right fist's centre: UP to the
  // guard, DOWN to the pommel (room for the left fist at -LEFT)
  const UP = 0.05, DOWN = 0.19, LEFT = 0.092, BLADE = 0.74, W = 0.052, T = 0.012;
  const add = (geo, m) => { const o = new THREE.Mesh(geo, m); o.frustumCulled = false; o.userData.partPick = pick(m.userData.part); root.add(o); return o; };
  // blade: a flat diamond section tapering to a point, extruded along +Z
  {
    const s = new THREE.Shape();
    s.moveTo(-W / 2, 0); s.lineTo(0, T / 2); s.lineTo(W / 2, 0); s.lineTo(0, -T / 2); s.closePath();
    add(new THREE.ExtrudeGeometry(s, { depth: BLADE * 0.86, bevelEnabled: false }), steel).position.z = UP + 0.024;
    const tip = new THREE.ConeGeometry(W / 2, BLADE * 0.14, 4, 1);
    tip.rotateX(Math.PI / 2); tip.scale(1, T / W, 1);
    add(tip, edge).position.z = UP + 0.024 + BLADE * 0.93;
    add(new THREE.BoxGeometry(W * 0.16, T * 1.08, BLADE * 0.7), edge).position.z = UP + 0.024 + BLADE * 0.38;   // the bright fuller
  }
  // cross guard, its quillons curling slightly toward the blade, with ball ends
  {
    const g = new THREE.BoxGeometry(0.2, 0.018, 0.022);
    const p = g.attributes.position;
    for (let i = 0; i < p.count; i++) { const x = p.getX(i); p.setZ(i, p.getZ(i) + 0.25 * x * x); }
    g.computeVertexNormals();
    add(g, gold).position.z = UP + 0.012;
    for (const x of [-0.1, 0.1]) add(new THREE.SphereGeometry(0.014, 12, 8), gold).position.set(x, 0, UP + 0.015);
  }
  // grip, wrapped (a lathe with ridges), and the pommel
  {
    const pts = [];
    for (let i = 0; i <= 32; i++) { const z = i / 32; pts.push(new THREE.Vector2(0.0145 + 0.002 * Math.abs(Math.sin(z * Math.PI * 11)), -DOWN + 0.01 + z * (UP + DOWN - 0.01))); }
    const g = new THREE.LatheGeometry(pts, 14);
    g.rotateX(Math.PI / 2);
    add(g, wrap);
    const pommel = new THREE.SphereGeometry(0.024, 14, 10);
    pommel.scale(1, 0.8, 1.15);
    add(pommel, gold).position.z = -DOWN - 0.008;
  }

  const curl = (side, w) => {   // a fist, by w
    const s = side === 'right' ? 1 : -1;
    for (const f of FINGERS) {
      bone(`${side}${f}Proximal`)?.rotation.set(0, 0, s * 1.35 * w);
      bone(`${side}${f}Intermediate`)?.rotation.set(0, 0, s * 1.5 * w);
      bone(`${side}${f}Distal`)?.rotation.set(0, 0, s * 0.9 * w);
    }
    bone(`${side}ThumbProximal`)?.rotation.set(0.5 * w, s * 0.35 * w, s * 0.3 * w);
    bone(`${side}ThumbIntermediate`)?.rotation.set(0, s * 0.5 * w, s * 0.35 * w);
    bone(`${side}ThumbDistal`)?.rotation.set(0, s * 0.4 * w, s * 0.3 * w);
  };
  // a fist's grip axis (little knuckle -> index knuckle) and centre, in the hand's own frame: the centre
  // of the curled fingers' joints and tips (where the grip sits in this skeleton's own fist)
  const fistOf = side => {
    const hand = bone(`${side}Hand`), keep = [];
    hand.traverse(o => keep.push([o, o.quaternion.clone()]));
    curl(side, 1);
    hand.updateMatrixWorld(true);
    const pts = [], at = V();
    for (const f of FINGERS) {
      const js = ['Proximal', 'Intermediate', 'Distal'].map(k => bone(`${side}${f}${k}`)?.getWorldPosition(V())).filter(Boolean);
      if (js.length === 3) pts.push(...js, js[2].clone().sub(js[1]).multiplyScalar(0.8).add(js[2]));
    }
    for (const p of pts) at.add(hand.worldToLocal(p));
    at.divideScalar(Math.max(1, pts.length));
    for (const [o, q] of keep) o.quaternion.copy(q);
    hand.updateMatrixWorld(true);
    const i = bone(`${side}IndexProximal`).position, l = bone(`${side}LittleProximal`).position;
    return { at, axis: V().subVectors(i, l).normalize() };
  };
  vrm.scene.updateMatrixWorld(true);
  const fistR = fistOf('right'), fistL = fistOf('left');
  root.position.copy(fistR.at);
  root.quaternion.setFromUnitVectors(V().set(0, 0, 1), fistR.axis);
  bone('rightHand').add(root);

  // turn a bone by a world-space rotation q
  const turn = (b, q) => {
    const pw = b.parent.getWorldQuaternion(Q());
    b.quaternion.premultiply(pw.clone().invert().multiply(q).multiply(pw));
    b.updateMatrixWorld(true);
  };
  const a = V(), e = V(), c = V(), u = V(), v = V(), n = V(), qa = Q();
  // two-bone IK: upper arm A, forearm B, hand C; the wrist to t, the elbow staying in its plane
  function reach(A, B, C, t) {
    A.getWorldPosition(a); B.getWorldPosition(e); C.getWorldPosition(c);
    const l1 = a.distanceTo(e), l2 = e.distanceTo(c);
    const d = Math.min(l1 + l2 - 1e-4, Math.max(Math.abs(l1 - l2) + 1e-4, a.distanceTo(t)));
    u.subVectors(a, e); v.subVectors(c, e);
    n.crossVectors(u, v);
    if (n.lengthSq() < 1e-10) n.crossVectors(u, V().set(0, -1, 0.3));   // straight arm: bend it downward
    n.normalize();
    const now = u.angleTo(v), want = Math.acos(Math.min(1, Math.max(-1, (l1 * l1 + l2 * l2 - d * d) / (2 * l1 * l2))));
    turn(B, qa.setFromAxisAngle(n, want - now));
    C.getWorldPosition(c);
    turn(A, qa.setFromUnitVectors(u.subVectors(c, a).normalize(), v.subVectors(t, a).normalize()));
  }

  const L = { up: null, fore: null, hand: null }, gripAt = V(), fistAt = V(), axisW = V(), handQ = Q(), want = V(), wrist = V(), fix = Q(), target = Q();
  const fistRW = V(), dir = V(), turnQ = Q(), rootW = Q(), smooth = x => { x = Math.min(1, Math.max(0, x)); return x * x * (3 - 2 * x); };
  let top = null, R = null;
  function grip() {
    if (!root.visible) return;
    curl('right', 1);
    L.up ||= bone('leftUpperArm'); L.fore ||= bone('leftLowerArm'); L.hand ||= bone('leftHand');
    if (!top) { top = L.hand; while (top.parent) top = top.parent; }
    R ||= bone('rightHand');
    top.updateMatrixWorld(true);
    // two-handed when the mocap's fists are close and the left one sits below the right along the grip
    root.getWorldPosition(fistRW);
    axisW.set(0, 0, 1).transformDirection(root.matrixWorld);
    fistAt.copy(fistL.at).applyMatrix4(L.hand.matrixWorld);
    dir.subVectors(fistRW, fistAt);
    const apart = dir.length(); dir.divideScalar(Math.max(apart, 1e-6));
    const w = smooth((FAR - apart) / (FAR - NEAR)) * smooth(dir.dot(axisW) / 0.4);
    stats = { w, apart, before: Math.acos(Math.min(1, Math.max(-1, dir.dot(axisW)))) * 180 / Math.PI };
    if (w <= 0) return;   // one-handed: the right fist's hilt, the mocap's left hand
    // the right wrist turns the hilt onto the line from the left fist through the right, as far as w
    // (a few times: turning the wrist moves the fist, and with it the line)
    for (let k = 0; k < 5; k++) {
      turnQ.setFromUnitVectors(axisW, dir).slerp(Q(), 1 - w);
      turn(R, turnQ);
      root.getWorldPosition(fistRW);
      axisW.set(0, 0, 1).transformDirection(root.matrixWorld);
      dir.subVectors(fistRW, fistAt).normalize();
    }
    // the left fist's place on the grip
    gripAt.set(0, 0, -LEFT).applyMatrix4(root.matrixWorld);
    axisW.set(0, 0, 1).transformDirection(root.matrixWorld);
    // turn the hand so its fist lies along the grip (index toward the blade), as far as w
    L.hand.getWorldQuaternion(handQ);
    target.setFromUnitVectors(fistL.axis.clone().applyQuaternion(handQ), axisW).multiply(handQ);
    handQ.slerp(target, w);
    // the wrist where that fist lands on the grip, then the arm to it
    want.copy(fistAt).lerp(gripAt, w);
    const s = L.hand.getWorldScale(V()).x;
    wrist.copy(fistL.at).multiplyScalar(s).applyQuaternion(handQ).negate().add(want);
    reach(L.up, L.fore, L.hand, wrist);
    // the hand's world rotation, under the moved arm
    L.hand.quaternion.copy(L.hand.parent.getWorldQuaternion(fix).invert().multiply(handQ));
    curl('left', 0.3 + 0.7 * w);
    // how well the hilt now sits between the hands (sword.stats, for checks)
    L.hand.updateMatrixWorld(true);
    want.copy(fistL.at).applyMatrix4(L.hand.matrixWorld);
    dir.subVectors(fistRW, want).normalize();
    stats.after = Math.acos(Math.min(1, Math.max(-1, dir.dot(axisW)))) * 180 / Math.PI;
    stats.leftOff = want.distanceTo(gripAt);
  }
  let stats = null;
  const pommelAt = V().set(0, 0, -DOWN - 0.035), tipAt = V().set(0, 0, UP + 0.024 + BLADE);
  function ends(p, q) { root.updateWorldMatrix(true, false); p.copy(pommelAt).applyMatrix4(root.matrixWorld); q.copy(tipAt).applyMatrix4(root.matrixWorld); }
  return { root, grip, ends, show(on) { root.visible = on; }, get shown() { return root.visible; }, get stats() { return stats; } };
}
