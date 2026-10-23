# Shader checkpoints

Everything the RunPod training pod produced, saved before it was shut down. `crisp.pt` and `light.pt`
are here (float16; load with `torch.load`, then `.float()` the tensors); the full set of 14 runs (float32
checkpoints, previews) and 12 exports was downloaded to `out/pod-backup/` in the session that made them,
which is not in git. `logs/` has every run's training log. The datasets are rebuilt by
`coverage/shader/build_dataset.py` from the renders and drawings (see `coverage/README.md`).

| run | what | last evaluation |
|---|---|---|
| `eval_sharp` | zero-iteration evaluation of sharp_lora (metrics only) | it 0  posed held-out L1 0.0133  film held-out L1 0.0152  shift error posed 0.0132 film 0.0141  line error / edge strength posed 0.1574 / 0.49 film 0.1989 / 0.49  2s |
| `hybrid_v1` | neural texture + U-Net decoder, wave 1 film | it 6000  loss 0.0259  held-out L1 0.0102  650s |
| `hybrid_v2` | hybrid on the full anchored film (Detail: Full) | it 12000  loss 0.0402  held-out L1 0.0110  727s |
| `mlp_v1` | first MLP decoder (baseline, superseded) | it 6000  loss 0.0382  held-out L1 0.0162  550s |
| `sharp_crisp` | sharp_equi + baked unsharp mask of its own drawing (Detail: Crisp, the default) | it 2000  posed held-out L1 0.0125  film held-out L1 0.0175  shift error posed 0.0020 film 0.0023  line error / edge strength posed 0.2566 / 0.62 film 0.3448 / 0.57  171s |
| `sharp_equi` | sharp_lora + shift-consistency loss against flicker (Detail: Smooth) | it 3000  posed held-out L1 0.0123  film held-out L1 0.0172  shift error posed 0.0024 film 0.0027  443s |
| `sharp_gan` | sharp_v1 + PatchGAN (not used) | it 5000  loss 0.0489  held-out L1 0.0133  481s |
| `sharp_light` | crisp + 7 light channels trained on 157 lit Hy Image drawings (lighting, not wired in yet) | it 4000  lit held-out L1 0.0219 (base, unlit: 0.0333)  film vs base 0.0004  398s |
| `sharp_lines` | linework try 1: target ink loss (dimmed the palette, not used) | it 2000  posed held-out L1 0.0217  film held-out L1 0.0230  shift error posed 0.0054 film 0.0058  line error / edge strength posed 0.0889 / 0.44 film 0.1197 / 0.39  316s |
| `sharp_lines2` | linework try 2: + keep-to-base (still dimmed, not used) | it 2000  posed held-out L1 0.0200  film held-out L1 0.0219  shift error posed 0.0047 film 0.0050  line error / edge strength posed 0.0923 / 0.46 film 0.1265 / 0.41  247s |
| `sharp_lines3` | linework try 3: tight ink mask, frozen texture (darkened, not used) | it 2000  posed held-out L1 0.0244  film held-out L1 0.0266  shift error posed 0.0063 film 0.0067  line error / edge strength posed 0.1004 / 0.59 film 0.1470 / 0.51  186s |
| `sharp_lora` | sharp + LoRA on posed mocap frames: hands/shoulders fixed (Detail: Sharp) | it 3000  posed held-out L1 0.0133  film held-out L1 0.0152  298s |
| `sharp_v1` | slim decoder at half res + full-res head (Sharp, before posed frames) | it 6000  loss 0.0791  held-out L1 0.0130  454s |
| `slim_v1` | slim decoder 16/32/48/64 (Detail: Fast) | it 12000  loss 0.0527  held-out L1 0.0123  626s |
