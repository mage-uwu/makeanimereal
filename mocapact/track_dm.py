"""Run the multi-clip policy in dm_control's own tracking task, as MoCapAct evaluates it.

    python mocapact/track_dm.py mocapact/npmp.npz CMU_016_22 [CMU_009_12 ...]

    python mocapact/track_dm.py mocapact/npmp.npz --subset locomotion_small 24

For each clip, from its first frame: control steps survived out of the clip's length (the episode
ends when the body strays from the reference, dm_control's termination error 0.3, or 9 steps
before the clip ends), and mean reward.
"""
import sys

import numpy as np
from dm_control import composer
from dm_control.locomotion.arenas import floors
from dm_control.locomotion.mocap import cmu_mocap_data
from dm_control.locomotion.tasks.reference_pose import tracking, types
from dm_control.locomotion.walkers import cmu_humanoid

from npmp import Npmp


def make_env(clip, ref_steps):
    task = tracking.MultiClipMocapTracking(
        cmu_humanoid.CMUHumanoidPositionControlledV2020, floors.Floor(size=(8., 8.)),
        ref_path=cmu_mocap_data.get_path_for_cmu(version='2020'), dataset=types.ClipCollection(ids=[clip]),
        ref_steps=tuple(ref_steps), reward_type='comic', min_steps=9, always_init_at_clip_start=True,
        termination_error_threshold=0.3)
    return composer.Environment(task=task, random_state=0)


def run(args):
    path, clip = args
    policy = Npmp(path)
    env = make_env(clip, policy.meta['ref_steps'])
    ts, z = env.reset(), policy.initial()
    length = len(env.task._clip_reference_features['joints']) - 1 - 9
    steps, rewards = 0, []
    while not ts.last():
        a, z = policy.act(ts.observation, z)
        ts = env.step(a)
        rewards.append(ts.reward)
        steps += 1
    return clip, steps, length, float(np.mean(rewards))


if __name__ == '__main__':
    import multiprocessing
    clips = sys.argv[2:]
    if clips[0] == '--subset':
        from dm_control.locomotion.tasks.reference_pose import cmu_subsets
        ids = cmu_subsets.CMU_SUBSETS_DICT[clips[1]].ids
        clips = sorted(set(ids))[::max(1, len(set(ids)) // int(clips[2]))][:int(clips[2])]
    done = []
    with multiprocessing.Pool(4) as pool:
        for clip, steps, length, r in pool.imap(run, [(sys.argv[1], c) for c in clips]):
            done.append(min(1, steps / length))
            print(f'{clip}: {steps} of {length} steps, reward {r:.3f}{"" if steps >= length else "  (fell or strayed)"}', flush=True)
    print(f'{len(done)} clips: mean fraction tracked {np.mean(done):.2f}, completed {np.mean(np.array(done) >= 1):.0%}')
