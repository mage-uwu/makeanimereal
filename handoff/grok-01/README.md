# Grok Imagine work order 01: skin the 3D walk with our character

For each angle, Grok gets two inputs:

- **Motion video:** the 3D model's walk cycle rendered from that camera angle (3 loops, 3 s, magenta background). It sets the pose, timing, perspective and framing.
- **Character reference:** our finished anime drawing of her from that same angle. It sets her look.

The text is only an instruction to apply the reference to the video. Nothing about her design is described in words.

| Angle | Motion video | Character reference | Save the result as |
|---|---|---|---|
| front (0°) | `y000_p15-motion.mp4` | `y000_p15-character.png` | `returns/y000_p15.mp4` |
| three-quarter (22.5°) | `y22.5_p15-motion.mp4` | `y22.5_p15-character.png` | `returns/y22.5_p15.mp4` |
| side (90°) | `y090_p15-motion.mp4` | `y090_p15-character.png` | `returns/y090_p15.mp4` |

`character-sheet.jpg` is the official character sheet. Add it as an extra reference if Grok accepts more than one image.

**Instruction (same for all three):**

> Restyle the video so the character is exactly the girl in the reference image: her design, colours and hand-drawn anime art style. Keep the video's motion, pose, camera, framing and magenta background exactly as they are.

When the three files are in `returns/`, tell me. I'll slice them into frames, colour-check them against the drawings, and put them in the viewer next to the current walk to compare.
