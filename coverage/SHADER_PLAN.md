# Learned render shader: plan

**Goal.** A shader that takes the base 3D model rendered live (any camera, any pose from the rig)
and draws it in the 2D style of the coverage film, in real time on device.

**Training data.** The coverage film gives us pairs: each styled frame comes with the exact
camera it was made from, so we can re-render the model's geometry from that camera, pixel for
pixel.
- **Wave 1:** 960 frames, pitch −20° to 70°.
- **Bottom wave:** 480 frames, −20° to −88°.
- **Total:** 1,440 pairs, every yaw.

## 1. What the shader sees (the G-buffer)

Rendered per frame from the posed, skinned mesh, so the same inputs exist for any pose at
run time:

| Channel | Why |
|---|---|
| Raw MToon colour | The model's own palette and texture detail; the network restyles it rather than inventing it |
| View-space normal (3) | Shading bands, rim and silhouette lines all follow the normal |
| Depth, and depth discontinuities | Outlines live at depth jumps |
| UV (2) and part ID | Ties appearance to the surface (hair, skin, dress, sash, shoes); keys the neural texture |
| View direction / camera pitch | Anime drawings change with angle (hair highlights, line weight) |
| Coverage mask | Alpha |

## 2. Model: a hybrid of neural texture and a screen-space network

This borrows from INR/NeRF. The appearance is a learned function stored on the surface, and
on top of it sits a screen-space stage that can draw what a surface field can't.

1. **Neural texture (the INR part).** A small learned feature grid in the model's UV space:
   - **Format:** about 8 channels, multi-resolution like Instant-NGP, baked to a 1024² half-float texture for deployment.
   - **What it learns:** what each surface point looks like in this style. Because it lives on the surface, it moves with the skeleton, so rigging comes free.
   - **View dependence:** the view direction enters through spherical-harmonic encoding, as in NeRF.
2. **Screen-space decoder (the translation part).** A compact U-Net (16–32 channels, about 0.5–1 M parameters):
   - **Inputs:** the G-buffer plus the sampled neural features.
   - **Output:** the styled RGBA.
   - **Why it's needed:** it draws what a surface field can't, such as line weight, stroke shapes at silhouettes, and soft hair edges.
3. **Ideas taken from NeRF-in-the-wild (training only):**
   - **Per-frame appearance code:** the 1,440 targets aren't perfectly consistent with each other (anchors versus bridges, small style wobble). A per-frame code absorbs that. At run time we use the mean code, so the shader learns the consistent style, not the noise.
   - **Per-frame alignment field:** a small 2D offset field, also per frame, absorbs the few pixels by which a drawing drifts against the geometry.

## 3. Losses

- **L1 plus a perceptual loss** (VGG/LPIPS) on colour, and **L1 on alpha** (the keyed mask).
- **A line loss:** an XDoG line map of the target compared with one of the output, so linework gets explicit weight.
- **A temporal loss:**
  - **Pairs:** consecutive film frames, compared through the exact flow computed from the known cameras and geometry.
  - **Purpose:** penalises flicker directly.
- **Optional:** a small adversarial term late in training, for crisp lines.

## 4. Data split and checks

- **Hold-out:** every 10th frame is held out. We report how close the output gets to those unseen views (L1 and perceptual distance).
- **Flicker metric:** frame-to-frame change after compensating for motion, on a slow orbit.
- **Novel-pose check** (by eye, since there is no target): arms raised, a walk cycle, head turns. If it fails, we add a **pose wave**: a few posed orbit clips restyled the same way (anchors plus bridges), which teaches it pose variation.

## 5. Phases and cost

| Phase | Where | Cost |
|---|---|---|
| 1. G-buffer export for all 1,440 cameras (and flow between consecutive frames) | local, headless three.js | free |
| 2. Training: a per-pixel baseline (neural texture plus MLP only), then the hybrid | RunPod A6000, PyTorch + nvdiffrast | about 1 h each, about $1.20 |
| 3. Browser runtime: neural texture as a WebGL texture, decoder in WebGPU / onnxruntime-web; "live" viewer mode with orbit and pose | local | free |
| 4. Pose wave, if the novel-pose check fails | RunPod + OpenRouter anchors | about $1.50 |

The budget after the coverage film is about $8.80, which covers phases 2 and 4 with room for a
few training iterations.

**Run-time target:**
- **Desktop:** 512×896 at 30 fps or better.
- **Phones:** half resolution.
- **Download:** the neural texture plus decoder weights, under 15 MB.
