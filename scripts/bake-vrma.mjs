#!/usr/bin/env node
// Bake retargeted mocap clips (coverage/motion/retarget_bvh.py's motions JSON) into VRM Animation
// files (.vrma, VRMC_vrm_animation 1.0): one glTF binary per clip, playable on any VRM by
// @pixiv/three-vrm-animation, UniVRM or VRoid Hub.
//
//   node scripts/bake-vrma.mjs [--motions assets/motions/cmu.json] [--out web/moves] [--fps 30] [--only wave,walk]
//
// The clips are posed by coverage/live/motion.js, the same retargeter the viewers use, on a generic
// VRoid-proportioned T-pose rig (REST below) whose bones carry no rotation. That rig is the files'
// rest skeleton, so each rotation key is the bone's normalized-pose rotation, which is what a VRMA
// player applies; the hips keys are positions on that rig, which players scale by hip height.
// Each file holds exactly one loop: the clip's own end-to-start crossfade is baked in, so its last
// key equals its first and it repeats seamlessly.
import * as THREE from 'three';
import fs from 'node:fs/promises';
import path from 'node:path';
import { parseArgs } from 'node:util';
import { createMotion } from '../coverage/live/motion.js';

const { values: opt } = parseArgs({
  options: {
    motions: { type: 'string', default: 'assets/motions/cmu.json' },
    out: { type: 'string', default: 'web/moves' },
    fps: { type: 'string', default: '30' },
    only: { type: 'string' },
  },
});

// Generic rest pose: VRM 1.0 T-pose in metres, facing +Z with the character's left on +X. Right-side
// bones mirror the left. [bone, parent, position (world)].
const LEFT = [
  ['leftShoulder', 'upperChest', [0.02, 1.25, -0.025]], ['leftUpperArm', 'leftShoulder', [0.105, 1.24, -0.025]],
  ['leftLowerArm', 'leftUpperArm', [0.315, 1.24, -0.025]], ['leftHand', 'leftLowerArm', [0.515, 1.24, -0.025]],
  ['leftThumbMetacarpal', 'leftHand', [0.52, 1.23, -0.01]], ['leftThumbProximal', 'leftThumbMetacarpal', [0.54, 1.225, 0.01]],
  ['leftThumbDistal', 'leftThumbProximal', [0.56, 1.22, 0.025]],
  ...['Index', 'Middle', 'Ring', 'Little'].flatMap((f, k) => {
    const z = [-0.007, -0.02, -0.033, -0.045][k], x = 0.575 - 0.004 * k;
    return [[`left${f}Proximal`, 'leftHand', [x, 1.245, z]], [`left${f}Intermediate`, `left${f}Proximal`, [x + 0.035, 1.245, z]],
      [`left${f}Distal`, `left${f}Intermediate`, [x + 0.06, 1.245, z]]];
  }),
  ['leftUpperLeg', 'hips', [0.076, 0.845, 0]], ['leftLowerLeg', 'leftUpperLeg', [0.076, 0.51, -0.007]],
  ['leftFoot', 'leftLowerLeg', [0.076, 0.12, -0.03]], ['leftToes', 'leftFoot', [0.076, 0.06, 0.075]],
];
const REST = [
  ['hips', null, [0, 0.885, 0]], ['spine', 'hips', [0, 0.935, 0.015]], ['chest', 'spine', [0, 1.04, 0.018]],
  ['upperChest', 'chest', [0, 1.145, 0.004]], ['neck', 'upperChest', [0, 1.275, -0.034]], ['head', 'neck', [0, 1.347, -0.025]],
  ...LEFT,
  ...LEFT.map(([b, p, [x, y, z]]) => [b.replace('left', 'right'), p.replace('left', 'right'), [-x, y, z]]),
];

// A stand-in for a loaded VRM: just the normalized humanoid rig motion.js poses.
function makeRig() {
  const scene = new THREE.Object3D(), bones = {}, rest = {};
  for (const [name, parent, pos] of REST) {
    const b = new THREE.Object3D();
    b.name = name;
    const p = parent ? REST.find(r => r[0] === parent)[2] : [0, 0, 0];
    b.position.set(pos[0] - p[0], pos[1] - p[1], pos[2] - p[2]);
    (parent ? bones[parent] : scene).add(b);
    bones[name] = b;
    rest[name] = b.position.clone();
  }
  const humanoid = {
    getNormalizedBoneNode: n => bones[n] ?? null,
    resetNormalizedPose() { for (const n in bones) { bones[n].quaternion.identity(); bones[n].position.copy(rest[n]); } },
  };
  return { scene, bones, humanoid };
}

// glTF binary with one animation: [time] -> per bone rotation, plus the hips' translation.
function writeVrma(name, times, rot, hips, meta) {
  const nodes = REST.map(([b]) => ({ name: b }));
  const index = Object.fromEntries(REST.map(([b], i) => [b, i]));
  REST.forEach(([b, parent, pos], i) => {
    const p = parent ? REST[index[parent]][2] : [0, 0, 0];
    nodes[i].translation = pos.map((v, k) => +(v - p[k]).toFixed(5));
    if (parent) (nodes[index[parent]].children ??= []).push(i);
  });
  const chunks = [], bufferViews = [], accessors = [];
  let offset = 0;
  const add = (arr, type, minmax) => {
    const data = Buffer.from(new Float32Array(arr).buffer);
    chunks.push(data);
    bufferViews.push({ buffer: 0, byteOffset: offset, byteLength: data.length });
    offset += data.length;
    const comps = { SCALAR: 1, VEC3: 3, VEC4: 4 }[type];
    accessors.push({ bufferView: bufferViews.length - 1, componentType: 5126, count: arr.length / comps, type, ...minmax });
    return accessors.length - 1;
  };
  const input = add(times, 'SCALAR', { min: [times[0]], max: [times.at(-1)] });
  // Bones that never move (the fingers' fixed curl, a still upper chest) hold one key over the loop.
  let hold = null;
  const samplers = [], channels = [];
  for (const [bone, values] of Object.entries(rot)) {
    const still = values.every((v, i) => Math.abs(v - values[i % 4]) < 1e-6);
    if (still) hold ??= add([times[0], times.at(-1)], 'SCALAR', { min: [times[0]], max: [times.at(-1)] });
    samplers.push({ input: still ? hold : input, output: add(still ? [...values.slice(0, 4), ...values.slice(0, 4)] : values, 'VEC4'), interpolation: 'LINEAR' });
    channels.push({ sampler: samplers.length - 1, target: { node: index[bone], path: 'rotation' } });
  }
  samplers.push({ input, output: add(hips, 'VEC3'), interpolation: 'LINEAR' });
  channels.push({ sampler: samplers.length - 1, target: { node: index.hips, path: 'translation' } });
  const bin = Buffer.concat(chunks);
  const json = {
    asset: { version: '2.0', generator: 'makeanimereal scripts/bake-vrma.mjs', copyright: meta.credit },
    extensionsUsed: ['VRMC_vrm_animation'],
    extensions: { VRMC_vrm_animation: { specVersion: '1.0', humanoid: { humanBones: Object.fromEntries(REST.map(([b], i) => [b, { node: i }])) } } },
    scene: 0, scenes: [{ nodes: [index.hips] }], nodes,
    animations: [{ name, channels, samplers }],
    buffers: [{ byteLength: bin.length }], bufferViews, accessors,
    extras: { source: meta.source },
  };
  const pad = (b, fill) => Buffer.concat([b, Buffer.alloc(-b.length & 3, fill)]);
  const js = pad(Buffer.from(JSON.stringify(json)), 0x20), bn = pad(bin, 0);
  const header = Buffer.alloc(12);
  header.writeUInt32LE(0x46546c67, 0); header.writeUInt32LE(2, 4); header.writeUInt32LE(12 + 8 + js.length + 8 + bn.length, 8);
  const chunk = (b, type) => { const h = Buffer.alloc(8); h.writeUInt32LE(b.length, 0); h.writeUInt32LE(type, 4); return Buffer.concat([h, b]); };
  return Buffer.concat([header, chunk(js, 0x4e4f534a), chunk(bn, 0x004e4942)]);
}

const motions = JSON.parse(await fs.readFile(opt.motions, 'utf8'));
const fps = Number(opt.fps);
const names = opt.only ? opt.only.split(',') : Object.keys(motions.clips);
await fs.mkdir(opt.out, { recursive: true });
const rig = makeRig();
const motion = createMotion(THREE, rig, motions);
const posed = REST.map(r => r[0]);
for (const name of names) {
  // One loop of the clip as motion.js plays it: its period is the clip's length minus the crossfade.
  const T = motion.duration(name), fade = Math.min(0.5, T / 3), period = T - fade;
  const n = Math.max(2, Math.round(period * fps));
  const times = [], rot = Object.fromEntries(posed.map(b => [b, []])), hips = [];
  const prev = {};
  for (let k = 0; k <= n; k++) {
    const t = (k / n) * period;
    rig.humanoid.resetNormalizedPose();
    motion.apply(name, t);
    times.push(+t.toFixed(5));
    for (const b of posed) {
      const q = rig.bones[b].quaternion.clone();
      if (prev[b] && prev[b].dot(q) < 0) q.set(-q.x, -q.y, -q.z, -q.w); // keep keys on one hemisphere
      prev[b] = q;
      rot[b].push(q.x, q.y, q.z, q.w);
    }
    hips.push(...rig.bones.hips.position.toArray());
  }
  const file = path.join(opt.out, `${name}.vrma`);
  await fs.writeFile(file, writeVrma(name, times, rot, hips, { credit: motions.credit, source: motions.clips[name].source }));
  console.log(`${file}: ${n + 1} keys, ${period.toFixed(2)} s loop`);
}
