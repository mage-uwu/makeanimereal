// A generic box character: a plain bone tree in the VRM 1.0 T-pose (facing +z), turned into a VRM
// humanoid by three-vrm's own VRMHumanoid, and dressed in outlined boxes. It has the same shape of
// API as a loaded VRM (scene, humanoid, meta, update), so every stage mode drives it unchanged.
// Its proportions are the generic rig the moves were baked on (scripts/bake-vrma.mjs), so they fit
// it exactly.
//
//   const vrm = createBoxCharacter(THREE, VRMHumanoid);

// [bone, parent, world position in metres].
const SIDE = [
  ['Shoulder', 'upperChest', [0.02, 1.25, -0.025]], ['UpperArm', 'Shoulder', [0.105, 1.24, -0.025]],
  ['LowerArm', 'UpperArm', [0.315, 1.24, -0.025]], ['Hand', 'LowerArm', [0.515, 1.24, -0.025]],
  ['UpperLeg', 'hips', [0.076, 0.845, 0]], ['LowerLeg', 'UpperLeg', [0.076, 0.51, -0.007]],
  ['Foot', 'LowerLeg', [0.076, 0.12, -0.03]], ['Toes', 'Foot', [0.076, 0.06, 0.075]],
];
const REST = [
  ['hips', null, [0, 0.885, 0]], ['spine', 'hips', [0, 0.935, 0.015]], ['chest', 'spine', [0, 1.04, 0.018]],
  ['upperChest', 'chest', [0, 1.145, 0.004]], ['neck', 'upperChest', [0, 1.275, -0.034]], ['head', 'neck', [0, 1.347, -0.025]],
  ...['left', 'right'].flatMap(side => SIDE.map(([b, p, [x, y, z]]) => [side + b, /^[A-Z]/.test(p) ? side + p : p, [side === 'left' ? x : -x, y, z]])),
];

// The boxes: [bone, size [x, y, z], centre (bone-local), colour key]; limbs run from the bone to its child.
const LIMBS = [['UpperArm', 'LowerArm', 0.085], ['LowerArm', 'Hand', 0.075], ['UpperLeg', 'LowerLeg', 0.12], ['LowerLeg', 'Foot', 0.1]];

export function createBoxCharacter(THREE, VRMHumanoid) {
  const scene = new THREE.Group();
  scene.name = 'BoxCharacter';
  const bones = {}, world = {};
  for (const [name, parent, pos] of REST) {
    const b = new THREE.Bone();
    b.name = name;
    world[name] = new THREE.Vector3(...pos);
    b.position.copy(world[name]);
    if (parent) b.position.sub(world[parent]);
    (parent ? bones[parent] : scene).add(b);
    bones[name] = b;
  }

  const css = getComputedStyle(document.documentElement);
  const token = (n, fallback) => css.getPropertyValue(n).trim() || fallback;
  const mats = {
    body: new THREE.MeshLambertMaterial({ color: '#ece6f0', emissive: '#6d6475' }),
    head: new THREE.MeshLambertMaterial({ color: '#f6f1f8', emissive: '#756c7d' }),
    accent: new THREE.MeshLambertMaterial({ color: token('--accent', '#e8507f'), emissive: '#7a2440' }),
    dark: new THREE.MeshBasicMaterial({ color: '#3a2a3f' }),
  };
  const edge = new THREE.LineBasicMaterial({ color: '#3a2a3f', transparent: true, opacity: 0.55 });
  const unit = new THREE.BoxGeometry(1, 1, 1), unitEdges = new THREE.EdgesGeometry(unit);
  const box = (bone, size, centre, mat, dir) => {
    const m = new THREE.Mesh(unit, mats[mat]);
    m.scale.set(...size);
    m.position.set(...centre);
    if (dir) m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    m.add(new THREE.LineSegments(unitEdges, edge));
    m.castShadow = true;
    bones[bone].add(m);
  };
  const local = (from, to) => world[to].clone().sub(world[from]);

  // Trunk and head.
  box('hips', [0.29, 0.17, 0.18], [0, -0.03, 0], 'body');
  box('spine', [0.26, 0.11, 0.16], [0, 0.05, 0], 'body');
  box('chest', [0.29, 0.11, 0.18], [0, 0.055, -0.005], 'body');
  box('upperChest', [0.33, 0.13, 0.19], [0, 0.065, -0.015], 'body');
  box('neck', [0.08, 0.08, 0.08], [0, 0.035, 0.005], 'body');
  box('head', [0.2, 0.23, 0.21], [0, 0.12, 0.02], 'head');
  box('head', [0.03, 0.035, 0.01], [-0.045, 0.13, 0.126], 'dark');            // eyes, on the face (+z)
  box('head', [0.03, 0.035, 0.01], [0.045, 0.13, 0.126], 'dark');
  // Limbs, along each bone to its child.
  for (const side of ['left', 'right']) {
    for (const [a, b, w] of LIMBS) {
      const d = local(side + a, side + b), len = d.length();
      box(side + a, [w, len * 0.96, w], d.clone().multiplyScalar(0.5).toArray(), 'body', d.clone().normalize());
    }
    const s = side === 'left' ? 1 : -1;
    box(side + 'Shoulder', [0.09, 0.08, 0.1], [s * 0.045, -0.005, 0], 'body');
    box(side + 'Hand', [0.1, 0.035, 0.08], [s * 0.05, 0, 0.005], 'accent');
    box(side + 'Foot', [0.095, 0.075, 0.17], [0, -0.075, 0.03], 'accent');
    box(side + 'Toes', [0.095, 0.05, 0.07], [0, -0.03, 0.03], 'accent');
  }

  scene.updateMatrixWorld(true);
  const humanoid = new VRMHumanoid(Object.fromEntries(Object.entries(bones).map(([n, node]) => [n, { node }])));
  scene.add(humanoid.normalizedHumanBonesRoot);
  return {
    scene, humanoid, expressionManager: null, lookAt: null,
    meta: { name: 'Box character', authors: ['makeanimereal'] },
    update() { humanoid.update(); },
  };
}
