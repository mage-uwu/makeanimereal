#!/usr/bin/env python3
"""Flag key drawings whose pose disagrees with the rig, using MediaPipe as an independent eye.

    python3 scripts/pose_check.py [--renders out/g2-renders] [--keys out/g2] [--web out/web2]
                                  [--cache out/cache2] [--model out/models/pose_landmarker_heavy.task]
                                  [--limit 40] [--report out/pose-check.json]

The rig's skeleton is the ground truth for every key: its joints are projected through the key's
camera and carried into the drawing by the drawing's fit (from build-web.py). MediaPipe's pose
model, trained on photos of people, reads arms and legs off the drawing itself. Where it finds a
pose, joints that land far from the rig's point to a drawing whose limbs were drawn elsewhere: a
redraw candidate. Only views with her face visible (within 100 degrees of the front, no steeper
than 45 degrees) are checked: elsewhere MediaPipe returns a pose that isn't hers. Those keys are
reported as unchecked rather than as passing.

Needs mediapipe (plus libegl1 on Linux) and the pose model:
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from latent import KeyCamera  # noqa: E402

# MediaPipe landmark -> Mixamo joint at the same place on the body.
JOINTS = {11: 'LeftArm', 12: 'RightArm', 13: 'LeftForeArm', 14: 'RightForeArm', 15: 'LeftHand', 16: 'RightHand',
          23: 'LeftUpLeg', 24: 'RightUpLeg', 25: 'LeftLeg', 26: 'RightLeg', 27: 'LeftFoot', 28: 'RightFoot'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--renders', default='out/g2-renders')
    p.add_argument('--keys', default='out/g2')
    p.add_argument('--web', default='out/web2')
    p.add_argument('--cache', default='out/cache2')
    p.add_argument('--model', default='out/models/pose_landmarker_heavy.task')
    p.add_argument('--limit', type=float, default=40, help='median joint error (px at 600 wide) that flags a drawing')
    p.add_argument('--report', default='out/pose-check.json')
    args = p.parse_args()

    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    manifest = json.loads((Path(args.renders) / 'manifest.json').read_text())
    mesh = json.loads((Path(args.renders) / 'mesh.json').read_text())
    keys = {Path(k['file']).stem: k for k in json.loads((Path(args.web) / 'keys.json').read_text())['keys']}
    bones = {b['name'].replace('mixamorig:', '').replace('mixamorig', ''): np.array(b['pos']) for b in mesh['skeleton']}
    cam0 = manifest['cameras'][0]
    detector = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.model), num_poses=1,
        min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3))

    results = []
    for v in manifest['views']:
        stem = Path(v['file']).stem
        if stem not in keys:
            continue
        k = keys[stem]
        cam = KeyCamera(v['yaw'], v['pitch'], cam0['target'], cam0['radius'], manifest['fov'], manifest['width'] / manifest['height'])
        im = Image.open(Path(args.keys) / v['file']).convert('RGB')
        W, H = im.size
        fit = np.load(Path(args.cache) / f'{stem}.fit.npy')
        fh, fw = fit.shape[:2]
        # MediaPipe reads people seen from the front half and at moderate heights; on back views
        # and steep top-down views it returns a pose, but not hers. Those stay unchecked.
        facing = min(v['yaw'] % 360, 360 - v['yaw'] % 360) <= 100 and v['pitch'] <= 45
        found = facing and detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(im))).pose_landmarks
        entry = {'file': v['file'], 'yaw': v['yaw'], 'pitch': v['pitch'], 'checked': bool(found)}
        if found:
            rig = {}
            for i, name in JOINTS.items():
                fx, fy, _ = cam.project(bones[name][None])
                fx, fy = float(fx[0]), float(fy[0])
                gx, gy = int(np.clip(fx * fw, 0, fw - 1)), int(np.clip(fy * fh, 0, fh - 1))
                fx, fy = fx + fit[gy, gx, 0] / fw, fy + fit[gy, gx, 1] / fh  # render -> placed drawing
                rig[i] = ((fx - k['ox']) / k['s'] * W, (fy - k['oy']) / k['s'] * H)
            lms = found[0]

            def errors(swap):
                # MediaPipe assumes a person facing it; seen from behind, its left and right swap.
                other = {11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15, 23: 24, 24: 23, 25: 26, 26: 25, 27: 28, 28: 27}
                return {JOINTS[i]: round(float(np.hypot(lms[other[i] if swap else i].x * W - rig[i][0],
                                                         lms[other[i] if swap else i].y * H - rig[i][1]) / W * 600), 1)
                        for i in JOINTS}
            errs = min(errors(False), errors(True), key=lambda e: np.median(list(e.values())))
            entry['joint_error'] = errs
            entry['median_error'] = float(np.median(list(errs.values())))
            entry['flag'] = entry['median_error'] > args.limit
        results.append(entry)

    checked = [r for r in results if r['checked']]
    flagged = sorted((r for r in checked if r['flag']), key=lambda r: -r['median_error'])
    Path(args.report).write_text(json.dumps(results, indent=1))
    print(f'{len(checked)} of {len(results)} drawings checked by MediaPipe; median joint error '
          f'{np.median([r["median_error"] for r in checked]):.1f}px; {len(flagged)} flagged over {args.limit:g}px')
    for r in flagged:
        print(f'  {r["file"]}  {r["median_error"]:.1f}px')
    print(f'report -> {args.report}')
    detector.close()


if __name__ == '__main__':
    main()
