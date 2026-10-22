// Motion capture, retargeted onto a VRM. Shared by the live viewer and the training renderer
// (coverage/orbit-page.html), so the shader trains on exactly the poses the viewer plays.
//
// motions.json (coverage/motion/retarget_bvh.py) holds mocap clips as, per bone and frame, the world
// direction of the bone and of a reference axis across it (the body's left, or for the arms its
// forward). The same frames are built here from the model's rest skeleton; a bone's world rotation is
// then (clip frame) x (rest frame)^-1, and its local one that relative to the nearest posed ancestor.
//
//   const motion = createMotion(THREE, vrm, motions);   // before posing the model at all
//   motion.apply('bye', 1.5);                           // pose the normalized bones for clip time 1.5 s

// Where each bone's direction is measured to (VRM rest), matching the BVH convention: to the child,
// and for the end bones (head, hands, toes) continuing from the parent.
const AIM = {
  hips: ['hips', 'spine'], spine: ['spine', 'chest'], chest: ['chest', 'neck'], neck: ['neck', 'head'], head: ['neck', 'head'],
  leftShoulder: ['leftShoulder', 'leftUpperArm'], leftUpperArm: ['leftUpperArm', 'leftLowerArm'], leftLowerArm: ['leftLowerArm', 'leftHand'], leftHand: ['leftLowerArm', 'leftHand'],
  rightShoulder: ['rightShoulder', 'rightUpperArm'], rightUpperArm: ['rightUpperArm', 'rightLowerArm'], rightLowerArm: ['rightLowerArm', 'rightHand'], rightHand: ['rightLowerArm', 'rightHand'],
  leftUpperLeg: ['leftUpperLeg', 'leftLowerLeg'], leftLowerLeg: ['leftLowerLeg', 'leftFoot'], leftFoot: ['leftFoot', 'leftToes'], leftToes: ['leftFoot', 'leftToes'],
  rightUpperLeg: ['rightUpperLeg', 'rightLowerLeg'], rightLowerLeg: ['rightLowerLeg', 'rightFoot'], rightFoot: ['rightFoot', 'rightToes'], rightToes: ['rightFoot', 'rightToes'],
};
const FINGERS = ['Thumb', 'Index', 'Middle', 'Ring', 'Little'].flatMap(f => ['Proximal', 'Intermediate', 'Distal'].map(s => f + s));
// The loop crossfades its last FADE seconds into its start.
const FADE = 0.5;

export function createMotion(THREE, vrm, motions) {
  const bone = n => vrm.humanoid.getNormalizedBoneNode(n);
  const rest = {};   // bone -> { q: rest frame (inverted), parent: nearest posed ancestor }
  const toModel = new THREE.Quaternion();
  const frameQ = (d, s, out) => {
    const x = new THREE.Vector3(...d).normalize(), y = new THREE.Vector3(...s);
    y.addScaledVector(x, -y.dot(x)).normalize();
    return out.setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, y, new THREE.Vector3().crossVectors(x, y)));
  };

  // Rest frames, from the model's rest skeleton.
  vrm.humanoid.resetNormalizedPose();
  vrm.scene.updateMatrixWorld(true);
  const pos = n => { const b = bone(n); return b && vrm.scene.worldToLocal(b.getWorldPosition(new THREE.Vector3())); };
  const left = pos('leftUpperLeg').sub(pos('rightUpperLeg')).setY(0).normalize(), up = new THREE.Vector3(0, 1, 0);
  const fwd = new THREE.Vector3().crossVectors(left, up);
  // The clips face +Z with the actor's left on +X; the model's own left and forward.
  toModel.setFromRotationMatrix(new THREE.Matrix4().makeBasis(left, up, fwd));
  for (const [name, [a, b]] of Object.entries(AIM)) {
    if (!bone(name) || !bone(a) || !bone(b)) continue;
    const ref = /Shoulder|Arm|Hand/.test(name) ? fwd : left;
    rest[name] = { q: frameQ(pos(b).sub(pos(a)).toArray(), ref.toArray(), new THREE.Quaternion()).invert() };
  }
  const posed = Object.keys(rest);
  for (const name of posed) {
    let p = bone(name).parent;
    while (p && !posed.some(n => bone(n) === p)) p = p.parent;
    rest[name].parent = p ? posed.find(n => bone(n) === p) : null;
  }
  const legLen = pos('leftUpperLeg').distanceTo(pos('leftLowerLeg')) + pos('leftLowerLeg').distanceTo(pos('leftFoot'));
  const hipsRest = bone('hips').position.clone();

  // One clip frame -> { bone: local quaternion }, hips position.
  function clipFrame(clip, f) {
    const B = motions.bones, world = {}, local = {};
    const q = new THREE.Quaternion();
    for (let k = 0; k < B.length; k++) {
      if (!rest[B[k]]) continue;
      const o = (f * B.length + k) * 6, a = clip.axes;
      frameQ([a[o], a[o + 1], a[o + 2]], [a[o + 3], a[o + 4], a[o + 5]], q);
      world[B[k]] = toModel.clone().multiply(q).multiply(rest[B[k]].q);
    }
    for (const n of Object.keys(world)) {
      const p = rest[n].parent;
      local[n] = p ? world[p].clone().invert().multiply(world[n]) : world[n];
    }
    const h = clip.hips, v = new THREE.Vector3(h[f * 3], 0, h[f * 3 + 2]).applyQuaternion(toModel).multiplyScalar(legLen);
    const hips = hipsRest.clone().add(v);
    hips.y += (h[f * 3 + 1] - motions.stand) * legLen;
    return { local, hips };
  }

  // Clip time t (s) -> pose, looping.
  function clipPose(clip, t) {
    const fps = motions.fps, T = (clip.frames - 1) / fps, fade = Math.min(FADE, T / 3);
    const u = fade + ((t % (T - fade)) + (T - fade)) % (T - fade);
    const at = time => {
      const x = Math.min(clip.frames - 1, time * fps), f0 = Math.floor(x), f1 = Math.min(clip.frames - 1, f0 + 1), w = x - f0;
      const A = clipFrame(clip, f0), B = clipFrame(clip, f1);
      for (const n in A.local) A.local[n].slerp(B.local[n], w);
      A.hips.lerp(B.hips, w);
      return A;
    };
    const P = at(u);
    if (u > T - fade) {
      const w = (u - (T - fade)) / fade, S = at(u - T + fade), e = w * w * (3 - 2 * w);
      for (const n in P.local) P.local[n].slerp(S.local[n], e);
      P.hips.lerp(S.hips, e);
    }
    return P;
  }

  // Pose the model for clip `name` at time t; false if there is no such clip.
  function apply(name, t) {
    const clip = motions.clips[name];
    if (!clip) return false;
    const P = clipPose(clip, t);
    for (const n in P.local) bone(n).quaternion.copy(P.local[n]);
    bone('hips').position.copy(P.hips);
    if (bone('upperChest')) bone('upperChest').quaternion.identity();
    // The mocap has no fingers: a relaxed curl.
    for (const side of ['left', 'right']) for (const f of FINGERS) {
      const b = bone(side + f);
      if (!b) continue;
      const c = f.startsWith('Thumb') ? 0.1 : f.endsWith('Proximal') ? 0.35 : 0.3;
      b.rotation.set(0, 0, side === 'left' ? -c : c);
    }
    return true;
  }

  return { apply, clips: Object.keys(motions.clips), duration: name => (motions.clips[name].frames - 1) / motions.fps };
}
