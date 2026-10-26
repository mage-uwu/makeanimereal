// Free run: PARC's motion generator drives the avatar live over a parkour course.
//
// The generator (parcgen.wasm in parcgen-worker.js) works in PARC's frame: z up, metres of PARC's
// humanoid. This module builds the course there (boxes rasterized into the heightfield the model
// sees, 0.2 m cells), steers it (WASD relative to the camera, or an auto-run lap), schedules the
// windows like PARC's KinematicController (each window seeds the next from its frames k, k + 1;
// k = 4 when the machine keeps up, later when it doesn't), and maps every frame onto the VRM:
// PARC (x fwd, y left, z up) -> VRM (z, x, y), scaled by the ratio of leg lengths.
//
//   const run = await createFreeRun({ THREE, scene, vrm, wasm, weights, start, physics });
//   each frame: run.update(dt, inputDir);  run.hips() for the camera;  run.dispose()
//
// With physics ({ mujoco, xml, weights, info }: tracker.js), the generator's frames are no longer
// shown: they are the reference a simulated body follows (MuJoCo, 30 Hz, the PPO tracker trained in
// physics/), and she is drawn from the simulation. Each new window is seeded where her simulated
// body actually is, so the plan follows her; if she falls past the tracker's limit she is put
// back on the plan (counted in the status line).

const FRAME = 78, FPS = 30;
// Sampling: DDIM over steps 20 and 0, mixed with a pass that cannot see the previous frames (CFG
// 0.65, PARC's default). Without the CFG pass the motion drifts into a crouch within ~15 s; at
// stride 20 a window costs ~80 ms in WASM, inside the 133 ms it covers (parcgen/simulate.mjs).
const STRIDE = 20, CFG = 0.65;
const PARC_LEG = 0.421546 + 0.40987;
// PARC body (index into joint rotations, 1-based like PARC's bodies) -> VRM bone.
const BONES = [[0, 'hips'], [1, 'spine'], [2, 'neck'], [3, 'rightUpperArm'], [4, 'rightLowerArm'], [5, 'rightHand'],
  [6, 'leftUpperArm'], [7, 'leftLowerArm'], [8, 'leftHand'], [9, 'rightUpperLeg'], [10, 'rightLowerLeg'], [11, 'rightFoot'],
  [12, 'leftUpperLeg'], [13, 'leftLowerLeg'], [14, 'leftFoot']];
const FINGERS = ['Thumb', 'Index', 'Middle', 'Ring', 'Little'].flatMap(f => ['Metacarpal', 'Proximal', 'Intermediate', 'Distal'].map(s => f + s));

// The course, in PARC's frame: a lap round a 24 m square, each side its own skill.
// Boxes are [x0, x1, y0, y1, height] in metres.
export const COURSE = {
  extent: 20, cell: 0.2,
  boxes: [
    // South side (heading +x along y = -12): walls to vault.
    [-6.3, -5.8, -15, -9, 0.9], [-0.3, 0.2, -15, -9, 1.0], [5.7, 6.2, -15, -9, 1.1],
    // East side (heading +y along x = 12): a step, a rooftop, a gap, another rooftop, a drop.
    [9, 15, -7, -5.5, 0.6], [9, 15, -5.5, 1.5, 1.4], [9, 15, 3.2, 8, 1.4],
    // North side (heading -x along y = 12): a tall block to climb and drop from.
    [3, 6, 9, 15, 2.2], [-4, -2, 9, 15, 0.8], [-6.5, -4, 9, 15, 1.6],
    // West side (heading -y along x = -12): stairs up and a drop.
    [-15, -9, 6, 7, 0.4], [-15, -9, 4.5, 6, 0.8], [-15, -9, 3, 4.5, 1.2], [-15, -9, -2, 3, 1.6],
    // The middle: scattered blocks.
    [-3, -1, -3, -1, 0.7], [2, 4, 1, 3, 1.3], [-1, 1, 4, 5, 0.5], [5, 6, -4, -2, 1.8],
  ],
  lap: [[-12, -12], [12, -12], [12, 12], [-12, 12]],
  spawn: [-14, -12],
};

export function rasterize(course) {
  const n = Math.round((2 * course.extent) / course.cell) + 1, heights = new Float32Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const x = -course.extent + i * course.cell, y = -course.extent + j * course.cell;
    let h = 0;
    for (const [x0, x1, y0, y1, bh] of course.boxes) if (x >= x0 && x <= x1 && y >= y0 && y <= y1 && bh > h) h = bh;
    heights[i * n + j] = h;
  }
  return { heights, nx: n, ny: n, minX: -course.extent, minY: -course.extent, cell: course.cell };
}

export async function createFreeRun({ THREE, scene, vrm, wasm, weights, start, physics = null, course = COURSE, onStatus = () => {} }) {
  // Her proportions: legs (for the scale) and where her hips sit above her legs.
  vrm.humanoid.resetNormalizedPose();
  vrm.scene.updateMatrixWorld(true);
  const at = n => vrm.humanoid.getNormalizedBoneNode(n).getWorldPosition(new THREE.Vector3());
  const leg = at('leftUpperLeg').distanceTo(at('leftLowerLeg')) + at('leftLowerLeg').distanceTo(at('leftFoot'));
  const k = leg / PARC_LEG;
  const hipsAboveLegs = at('hips').sub(at('leftUpperLeg').add(at('rightUpperLeg')).multiplyScalar(0.5));
  const toVrm = (x, y, z, out = new THREE.Vector3()) => out.set(y * k, z * k, x * k);

  // The course: the model's heightfield, and boxes to look at.
  const field = rasterize(course);
  const group = new THREE.Group();
  const stone = new THREE.MeshLambertMaterial({ color: '#d9cfd4', emissive: '#5f5660' });
  const boxes = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), stone, course.boxes.length);
  course.boxes.forEach(([x0, x1, y0, y1, h], i) => {
    const c = toVrm((x0 + x1) / 2, (y0 + y1) / 2, h / 2);
    boxes.setMatrixAt(i, new THREE.Matrix4().makeScale((y1 - y0) * k, h * k, (x1 - x0) * k).setPosition(c));
    boxes.setColorAt(i, new THREE.Color('#b9adb6').lerp(new THREE.Color('#efe6ea'), Math.min(1, h / 2.2)));
  });
  const groundSize = 2 * course.extent * k;
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(groundSize, groundSize), new THREE.MeshLambertMaterial({ color: '#e7dde2', emissive: '#6a6168' }));
  ground.rotation.x = -Math.PI / 2;
  const grid = new THREE.GridHelper(groundSize, 2 * course.extent / 2, '#cbbfc6', '#d8ccd3');
  grid.position.y = 0.002;
  group.add(ground, grid, boxes);
  scene.add(group);

  // The generator, in its worker.
  const worker = new Worker(new URL('./parcgen-worker.js', import.meta.url));
  await new Promise((resolve, reject) => {
    worker.onmessage = ({ data }) => (data.type === 'ready' ? resolve() : reject(new Error(data.message)));
    worker.postMessage({ type: 'init', wasm, weights }, [wasm, weights]);
  });
  worker.postMessage({ type: 'terrain', ...field });

  // The first window: the standing frames, moved to the spawn point.
  const first = new Float32Array(15 * FRAME);
  for (let f = 0; f < 15; f++) {
    first.set(start[Math.min(f, 1)], f * FRAME);
    first[f * FRAME] += course.spawn[0]; first[f * FRAME + 1] += course.spawn[1];
  }
  let cur = { frames: first, k: 4 }, next = null, pending = 0, t = 2, avgMs = 60, lapIdx = 1, stopped = false;
  const inputDir = [0, 0];
  const frameAt = i => cur.frames.subarray(Math.min(14, i) * FRAME, Math.min(14, i) * FRAME + FRAME);

  // The simulated body, if physics is on: the last two simulated frames, for drawing in between.
  let tracker = null, acc = 0, falls = 0;
  const sim = new Float32Array(2 * FRAME);
  if (physics) {
    const { createTracker } = await import('./tracker.js');
    const ground = (x, y) => field.heights[Math.max(0, Math.min(field.nx - 1, Math.round((x - field.minX) / field.cell))) * field.ny +
      Math.max(0, Math.min(field.ny - 1, Math.round((y - field.minY) / field.cell)))];
    tracker = await createTracker({ ...physics, boxes: course.boxes, ground });
    tracker.reset(frameAt(2), frameAt(3));
    sim.set(tracker.frame(), 0); sim.set(tracker.frame(), FRAME);
  }

  const request = () => {
    // Seed from frames k, k + 1: late enough that the result arrives before frame k + 2 is due.
    const kk = Math.min(13, Math.max(4, Math.ceil(avgMs / (1000 / FPS)) + 1));
    cur.k = kk;
    const id = ++pending;
    const prev = cur.frames.slice(kk * FRAME, (kk + 2) * FRAME);
    if (tracker) {
      // Plan from where her body is: shift the seed by the simulation's offset from the plan now.
      const now = frameAt(Math.floor(t)), body = sim.subarray(FRAME);
      for (const o of [0, FRAME]) { prev[o] += body[0] - now[0]; prev[o + 1] += body[1] - now[1]; }
    }
    worker.postMessage({ type: 'generate', id, prev, target: inputDir.slice(), stride: STRIDE, cfg: CFG });
  };
  const status = () => onStatus(`${Math.round(avgMs)} ms per half second of motion` + (tracker ? ` · physics · ${falls} fall${falls === 1 ? '' : 's'}` : ''));
  worker.onmessage = ({ data }) => {
    if (data.type === 'error') { onStatus(`Generator error: ${data.message}`); return; }
    if (data.type !== 'window' || data.id !== pending) return;
    avgMs = avgMs * 0.8 + data.ms * 0.2;
    next = data.frames;
    status();
  };
  request();

  // Hand over to the next window once its seed frames have played: old k + 1 is new 1.
  const handOver = () => {
    if (next && t >= cur.k + 1) {
      t = Math.min(t - cur.k, 2);                                       // a late window starts near its top
      cur = { frames: next, k: 4 };
      next = null;
      request();
    }
  };

  // One 30 Hz physics tick: the body follows the plan's next frame (looking one further ahead).
  const tick = () => {
    const i = Math.floor(t);
    const { fell } = tracker.step(frameAt(i + 1), frameAt(i + 2));
    t = Math.min(14, i + 1);
    if (fell) { falls++; tracker.reset(frameAt(t), frameAt(t + 1)); status(); }
    sim.copyWithin(0, FRAME);
    sim.set(tracker.frame(), FRAME);
    handOver();
  };

  // One PARC frame (fractional index into a window) -> VRM normalized pose.
  const qa = new THREE.Quaternion(), qb = new THREE.Quaternion(), pa = new THREE.Vector3(), pb = new THREE.Vector3();
  const hipsWorld = new THREE.Vector3();
  function pose(frames, ft) {
    const last = frames.length / FRAME - 1;
    const f0 = Math.min(last, Math.floor(ft)), f1 = Math.min(last, f0 + 1), w = Math.min(1, ft - f0);
    const A = f0 * FRAME, B = f1 * FRAME;
    vrm.humanoid.resetNormalizedPose();
    for (const [b, name] of BONES) {
      const node = vrm.humanoid.getNormalizedBoneNode(name);
      if (!node) continue;
      const o = b === 0 ? 3 : 7 + 4 * (b - 1);
      // PARC quaternion (x, y, z, w) in VRM axes: the vector part permutes like positions.
      qa.set(frames[A + o + 1], frames[A + o + 2], frames[A + o], frames[A + o + 3]);
      qb.set(frames[B + o + 1], frames[B + o + 2], frames[B + o], frames[B + o + 3]);
      node.quaternion.copy(qa.slerp(qb, w));
    }
    for (const side of ['left', 'right']) for (const f of FINGERS) {
      const node = vrm.humanoid.getNormalizedBoneNode(side + f);
      if (!node) continue;
      const c = f.startsWith('Thumb') ? 0.1 : f.endsWith('Proximal') ? 0.35 : 0.3;
      node.rotation.set(0, 0, side === 'left' ? -c : c);
    }
    toVrm(frames[A], frames[A + 1], frames[A + 2], pa);
    toVrm(frames[B], frames[B + 1], frames[B + 2], pb);
    const hips = vrm.humanoid.getNormalizedBoneNode('hips');
    hips.position.copy(pa.lerp(pb, w)).add(hipsAboveLegs.clone().applyQuaternion(hips.quaternion));
    hipsWorld.copy(hips.position);
  }

  return {
    // dir: [x, y] wanted direction in PARC's ground plane, or null for the auto-run lap.
    update(dt, dir) {
      if (stopped) return;
      const root = tracker ? [sim[FRAME], sim[FRAME + 1]] : [cur.frames[Math.min(14, Math.floor(t)) * FRAME], cur.frames[Math.min(14, Math.floor(t)) * FRAME + 1]];
      if (dir) { inputDir[0] = dir[0]; inputDir[1] = dir[1]; }
      else {
        // Auto-run: head for the next corner of the lap.
        let [gx, gy] = course.lap[lapIdx];
        if (Math.hypot(gx - root[0], gy - root[1]) < 2.5) { lapIdx = (lapIdx + 1) % course.lap.length; [gx, gy] = course.lap[lapIdx]; }
        const d = Math.hypot(gx - root[0], gy - root[1]) || 1;
        inputDir[0] = (gx - root[0]) / d; inputDir[1] = (gy - root[1]) / d;
      }
      if (tracker) {
        acc += dt;
        for (let n = 0; acc >= 1 / FPS && n < 4; n++) { acc -= 1 / FPS; tick(); }
        acc = Math.min(acc, 1 / FPS);
        pose(sim, acc * FPS);                                             // between the last two physics frames
        return;
      }
      t = Math.min(14, t + dt * FPS);
      handOver();
      pose(cur.frames, t);
    },
    hips: () => hipsWorld,
    // Where she is in PARC's frame (for tests): root position at the current frame.
    root: () => (tracker ? Array.from(sim.slice(FRAME, FRAME + 3)) : Array.from(cur.frames.slice(Math.min(14, Math.floor(t)) * FRAME, Math.min(14, Math.floor(t)) * FRAME + 3))),
    falls: () => falls,
    ground: (x, y) => { const i = Math.round((x - field.minX) / field.cell), j = Math.round((y - field.minY) / field.cell); return field.heights[Math.max(0, Math.min(field.nx - 1, i)) * field.ny + Math.max(0, Math.min(field.ny - 1, j))]; },
    toVrm,
    scale: k,
    dispose() {
      stopped = true;
      worker.terminate();
      tracker?.dispose();
      scene.remove(group);
      boxes.geometry.dispose(); stone.dispose(); ground.geometry.dispose(); ground.material.dispose(); grid.dispose();
      vrm.humanoid.resetNormalizedPose();
    },
  };
}
