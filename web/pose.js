// Shared static poses. Imported by the offline renderer (scripts/render-page.html) and the web
// viewer (web/index.html) so both bend the skeleton identically: the viewer's mesh must match the
// renders the key drawings were made from, or the projection drifts.

// Turn `bone` (in world space) so the direction from it to `child` points along `dir`.
function aim(THREE, root, bone, child, dir) {
  root.updateMatrixWorld(true);
  const from = bone.getWorldPosition(new THREE.Vector3());
  const cur = child.getWorldPosition(new THREE.Vector3()).sub(from).normalize();
  const turn = new THREE.Quaternion().setFromUnitVectors(cur, dir.clone().normalize());
  const parentWorld = bone.parent.getWorldQuaternion(new THREE.Quaternion());
  const world = bone.getWorldQuaternion(new THREE.Quaternion()).premultiply(turn);
  bone.quaternion.copy(parentWorld.invert().multiply(world));
}

const find = (root, name) => root.getObjectByName(name) || root.getObjectByName(name.replace(':', ''));

// Arms hanging at the sides with a slight outward angle and a soft elbow bend, built from the
// T-pose by aiming the Mixamo arm bones rather than hard-coding rotations.
export function relaxArms(THREE, root) {
  const hips = find(root, 'mixamorig:Hips').getWorldPosition(new THREE.Vector3());
  for (const side of ['Left', 'Right']) {
    const arm = find(root, `mixamorig:${side}Arm`);
    const fore = find(root, `mixamorig:${side}ForeArm`);
    const hand = find(root, `mixamorig:${side}Hand`);
    if (!arm || !fore || !hand) throw new Error(`relaxed pose needs mixamorig:${side}Arm/ForeArm/Hand bones`);
    const out = Math.sign(arm.getWorldPosition(new THREE.Vector3()).x - hips.x);
    aim(THREE, root, arm, fore, new THREE.Vector3(out * 0.36, -1, 0.03));
    aim(THREE, root, fore, hand, new THREE.Vector3(out * 0.42, -1, 0.12));
  }
  root.updateMatrixWorld(true);
}
