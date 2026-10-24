// A longsword for her right hand: built from a few lathed and extruded pieces (no model file), held
// on the normalized right hand bone, whose axes are the same on any VRM: in the T-pose the fingers
// point along -X, the palm faces -Y and her front is +Z. The blade leaves the fist along the hand's +Z.
//
//   const sword = createSword({ THREE, vrm });   // after the model loads; hidden until shown
//   sword.show(true);                            // she draws it
//   sword.grip();                                // each frame, after the pose: curl the fingers round it
//   sword.ends(a, b);                            // its pommel and tip, world space (for the crop box)
//
// The network sees it like her other meshes: colour from its own flat materials, normals, depth and
// UVs from the geometry, and `part` (userData.partName) picks which part's neural texture it samples:
// her silver hair's for the steel, the shorts' (a colour-only part, so the network follows the colour)
// for the gold, her dress's for the navy grip.
const PART = { steel: 'N00_000_Hair_00_HAIR', gold: 'Shorts', navy: 'N00_010_01_Onepiece_00_CLOTH_01' };
export function createSword({ THREE, vrm }) {
  const hand = vrm.humanoid.getNormalizedBoneNode('rightHand');
  const root = new THREE.Group();
  root.name = 'sword';
  root.visible = false;
  const mat = (color, name, part) => Object.assign(new THREE.MeshBasicMaterial({ color, name }), { userData: { part } });
  const steel = mat(0xcfd6e2, 'SwordBlade', PART.steel), edge = mat(0xf2f5fa, 'SwordEdge', PART.steel), gold = mat(0xc9a24a, 'SwordGuard', PART.gold), wrap = mat(0x1c2056, 'SwordGrip', PART.navy);

  // lengths in metres, along +Z from the fist's centre
  const GRIP = 0.17, BLADE = 0.74, W = 0.052, T = 0.012;
  const parts = [];
  const add = (geo, m) => { const o = new THREE.Mesh(geo, m); o.frustumCulled = false; o.userData.partName = m.userData.part; root.add(o); parts.push(o); return o; };
  // blade: a flat diamond section tapering to a point, extruded along +Z
  {
    const s = new THREE.Shape();
    s.moveTo(-W / 2, 0); s.lineTo(0, T / 2); s.lineTo(W / 2, 0); s.lineTo(0, -T / 2); s.closePath();
    const g = new THREE.ExtrudeGeometry(s, { depth: BLADE * 0.86, bevelEnabled: false });
    add(g, steel).position.z = GRIP / 2 + 0.03;
    const tip = new THREE.ConeGeometry(W / 2, BLADE * 0.14, 4, 1);
    tip.rotateX(Math.PI / 2); tip.scale(1, T / W, 1);
    add(tip, edge).position.z = GRIP / 2 + 0.03 + BLADE * 0.86 + BLADE * 0.07;
    const fuller = new THREE.BoxGeometry(W * 0.16, T * 1.08, BLADE * 0.7);   // the bright groove down its middle
    add(fuller, edge).position.z = GRIP / 2 + 0.03 + BLADE * 0.38;
  }
  // cross guard, its quillons curling slightly toward the blade, and a ricasso block
  {
    const g = new THREE.BoxGeometry(0.2, 0.018, 0.022);
    const p = g.attributes.position;
    for (let i = 0; i < p.count; i++) { const x = p.getX(i); p.setZ(i, p.getZ(i) + 0.25 * x * x); }
    g.computeVertexNormals();
    add(g, gold).position.z = GRIP / 2 + 0.015;
    add(new THREE.SphereGeometry(0.014, 12, 8), gold).position.set(0.1, 0, GRIP / 2 + 0.018);
    add(new THREE.SphereGeometry(0.014, 12, 8), gold).position.set(-0.1, 0, GRIP / 2 + 0.018);
  }
  // grip, wrapped (a lathe with ridges), and the pommel
  {
    const pts = [];
    for (let i = 0; i <= 24; i++) { const z = i / 24; pts.push(new THREE.Vector2(0.0145 + 0.002 * Math.abs(Math.sin(z * Math.PI * 8)), (z - 0.5) * GRIP)); }
    const g = new THREE.LatheGeometry(pts, 14);
    g.rotateX(Math.PI / 2);
    add(g, wrap);
    const pommel = new THREE.SphereGeometry(0.024, 14, 10);
    pommel.scale(1, 0.8, 1.15);
    add(pommel, gold).position.z = -GRIP / 2 - 0.02;
  }
  // the fist: the grip runs across the palm, below it and toward the fingers
  root.position.set(-0.075, -0.03, 0.0);
  hand.add(root);

  const FINGERS = ['Index', 'Middle', 'Ring', 'Little'];
  function grip() {
    if (!root.visible) return;
    const bone = n => vrm.humanoid.getNormalizedBoneNode(n);
    for (const f of FINGERS) {
      bone(`right${f}Proximal`)?.rotation.set(0, 0, 1.35);
      bone(`right${f}Intermediate`)?.rotation.set(0, 0, 1.5);
      bone(`right${f}Distal`)?.rotation.set(0, 0, 0.9);
    }
    bone('rightThumbProximal')?.rotation.set(0.5, 0.35, 0.3);
    bone('rightThumbIntermediate')?.rotation.set(0, 0.5, 0.35);
    bone('rightThumbDistal')?.rotation.set(0, 0.4, 0.3);
  }
  const pommelAt = new THREE.Vector3(0, 0, -GRIP / 2 - 0.045), tipAt = new THREE.Vector3(0, 0, GRIP / 2 + 0.03 + BLADE);
  function ends(a, b) { root.updateWorldMatrix(true, false); a.copy(pommelAt).applyMatrix4(root.matrixWorld); b.copy(tipAt).applyMatrix4(root.matrixWorld); }
  return { root, parts, grip, ends, show(on) { root.visible = on; }, get shown() { return root.visible; } };
}
