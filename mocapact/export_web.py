"""MoCapAct's multi-clip policy for the browser: web/npmp/.

    python mocapact/export_web.py <multiclip_policy/full_dataset/model/model.ckpt> [out dir] [--fixture fixture.json]

Writes
  cmu.xml       dm_control's CMU humanoid (2020, position controlled, as MoCapAct trained it) on
                its floor, compiled by dm_control and exported as one MJCF file
  npmp.json     where each observable lives in MuJoCo's arrays, the network's layout, timesteps
  npmp.bin      the network (float16): normalizers, then each layer's weight (out x in) and bias
  clips/*.json  reference clips: per 30 Hz frame, qpos (63) and the 31 tracked bodies' positions
                and quaternions as dm_control's CMU data stores them
With --fixture, also a test fixture: an episode in dm_control (state, observations, actions per
step) for web/npmp.js to reproduce (mocapact/test_browser.mjs).
"""
import json
import os
import sys

import mujoco
import numpy as np
from dm_control import mjcf

from ckpt import load
from npmp import Npmp
from track_dm import make_env

CLIPS = {  # CMU clips the policy tracks to the end, from dm_control's CMU 2020 data
    'CMU_016_22': 'Walk', 'CMU_016_27': 'Walk, turn left', 'CMU_069_64': 'Walk sideways, turn',
    'CMU_016_44': 'Jog, turn right', 'CMU_127_09': 'Run', 'CMU_016_02': 'Jump',
    'CMU_083_45': 'Hop forward', 'CMU_091_44': 'Small jump',
}
TORQUE_THRESHOLD, TOUCH_THRESHOLD = 60, 1e-3  # dm_control's cmu_humanoid.py and legacy_base.py

args = [a for a in sys.argv[1:] if not a.startswith('--')]
src, out = args[0], (args[1] if len(args) > 1 else 'web/npmp')
fixture = sys.argv[sys.argv.index('--fixture') + 1] if '--fixture' in sys.argv else None
os.makedirs(os.path.join(out, 'clips'), exist_ok=True)

# The network.
ck = load(src)
hp, sd = ck['hyper_parameters'], ck['state_dict']
tmp = os.path.join(out, '.npmp.npz')
os.system(f'{sys.executable} -W ignore {os.path.join(os.path.dirname(__file__), "export.py")} {src} {tmp} > /dev/null')
policy = Npmp(tmp)
os.remove(tmp)
w = policy.w
blobs, layout = [], []


def put(name, a):
    a = np.ascontiguousarray(a, np.float32)
    layout.append([name, list(a.shape)])
    blobs.append(a.ravel())


for net in ('ref_encoder', 'decoder'):
    put(f'{net}_mean', w[f'{net}_mean'])
    put(f'{net}_std', w[f'{net}_std'])
nets = {'encoder': ('reference_encoder_encoder', [0, 3, 6]), 'decoder': ('action_decoder', [0, 3, 6, 9])}
for name, (prefix, idx) in nets.items():
    for i in idx:
        put(f'{name}.{i}.W', w[f'{prefix}_{i}_weight'])
        put(f'{name}.{i}.b', w[f'{prefix}_{i}_bias'])
        if i != idx[-1]:
            put(f'{name}.{i + 1}.g', w[f'{prefix}_{i + 1}_weight'])
            put(f'{name}.{i + 1}.beta', w[f'{prefix}_{i + 1}_bias'])
np.concatenate(blobs).astype(np.float16).tofile(os.path.join(out, 'npmp.bin'))

# The body: the compiled model, and where each observable is.
env = make_env('CMU_016_22', policy.meta['ref_steps'])
env.reset()
task, walker, ph = env.task, env.task._walker, env.physics
mjcf.export_with_assets(task.root_entity.mjcf_model, out, 'cmu.xml')
m = ph.model.ptr
sensor = lambda name: list(range(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, 'walker/' + name)],
                                 m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, 'walker/' + name)] +
                                 m.sensor_dim[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, 'walker/' + name)]))
sensors = lambda tag: [i for s in walker.mjcf_model.sensor.all_children() if s.tag == tag for i in sensor(s.name)]
info = dict(
    source='MoCapAct multi-clip policy (full dataset), microsoft/mocapact-models, CDLA-Permissive-2.0',
    observables=policy.meta['observables'], sizes=policy.meta['sizes'], ref_steps=policy.meta['ref_steps'],
    embed=policy.meta['embed_size'], layout=layout,
    timestep=float(m.opt.timestep), substeps=int(round(env.control_timestep() / m.opt.timestep)), control_dt=float(env.control_timestep()),
    nq=int(m.nq), nv=int(m.nv), nu=int(m.nu),
    joints_qpos=[int(ph.bind(j).qposadr) for j in walker.observable_joints],
    joints_dof=[int(ph.bind(j).dofadr) for j in walker.observable_joints],
    mocap_joints_qpos=[int(ph.bind(j).qposadr) for j in walker.mocap_joints],
    sensors_velocimeter=sensors('velocimeter'), sensors_gyro=sensors('gyro'), sensors_touch=sensors('touch'),
    sensors_torque=sensors('torque'), end_effectors_pos=sensors('framepos'),
    torque_threshold=TORQUE_THRESHOLD, touch_threshold=TOUCH_THRESHOLD,
    root=int(ph.bind(walker.root_body).element_id),
    bodies=[int(ph.bind(b).element_id) for b in walker.mocap_tracking_bodies],
    body_names=[b.name for b in walker.mocap_tracking_bodies],
    termination=0.3,
)
with open(os.path.join(out, 'npmp.json'), 'w') as f:
    json.dump(info, f, separators=(',', ':'))

# The clips.
index = []
for clip, label in CLIPS.items():
    e = make_env(clip, policy.meta['ref_steps'])
    e.reset()
    r = e.task._clip_reference_features
    T = len(r['joints'])
    qpos = np.zeros((T, m.nq))
    qpos[:, :3], qpos[:, 3:7] = r['position'], r['quaternion']
    qpos[:, info['mocap_joints_qpos']] = r['joints']
    frames = np.concatenate([qpos, r['body_positions'].reshape(T, -1), r['body_quaternions'].reshape(T, -1)], 1)
    with open(os.path.join(out, 'clips', clip + '.json'), 'w') as f:
        json.dump(dict(id=clip, label=label, dt=info['control_dt'], frames=np.round(frames, 5).tolist()), f, separators=(',', ':'))
    index.append(dict(id=clip, label=label, frames=T))
with open(os.path.join(out, 'clips', 'index.json'), 'w') as f:
    json.dump(index, f, indent=1)

# A fixture: one episode in dm_control, step by step.
if fixture:
    ts, z = env.reset(), policy.initial()
    steps = []
    for _ in range(40):
        d = env.physics.data
        state = dict(time=float(d.time), qpos=d.qpos.tolist(), qvel=d.qvel.tolist(), act=d.act.tolist(), ctrl=d.ctrl.tolist(),
                     sensordata=d.sensordata.tolist(), qacc_warmstart=d.qacc_warmstart.tolist(), time_step=int(env.task._time_step))
        enc, dec = policy.features(ts.observation, 'ref_encoder'), policy.features(ts.observation, 'decoder')
        a, z = policy.act(ts.observation, z)
        steps.append(dict(state=state, ref_encoder=enc.tolist(), decoder=dec.tolist(), z=z.tolist(), action=a.tolist()))
        ts = env.step(a)
        if ts.last():
            break
    with open(fixture, 'w') as f:
        json.dump(dict(clip='CMU_016_22', steps=steps), f)
print(f'{out}: {sum(b.size for b in blobs)} weights, {len(index)} clips' + (f', fixture {fixture}' if fixture else ''))
