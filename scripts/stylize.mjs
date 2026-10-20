#!/usr/bin/env node
// Turn every render from scripts/render.mjs into an anime illustration of the same view via an
// OpenRouter image model. The style reference image fixes the art style; the first stylized
// front view is fed back as an identity anchor so every angle draws the same character.
//
//   OPENROUTER_API_KEY=... node scripts/stylize.mjs [--renders out/renders] [--out out/stylized]
//        [--style assets/emerald-pixie-ref.jpg] [--style-only] [--model google/gemini-3.1-flash-lite-image] [--notes assets/emerald-pixie-notes.txt]
//        [--concurrency 4] [--only y000_p0.png,...] [--force] [--no-anchor] [--dry-run]
import fs from 'node:fs/promises';
import path from 'node:path';
import { parseArgs } from 'node:util';

const { values: opt } = parseArgs({
  options: {
    renders: { type: 'string', default: 'out/renders' },
    out: { type: 'string', default: 'out/stylized' },
    style: { type: 'string', default: 'assets/emerald-pixie-ref.jpg' },
    // By default the reference is a sheet of the same character; --style-only borrows just its art style.
    'style-only': { type: 'boolean', default: false },
    // Plain-text list of design details the model must keep at every angle (skipped if missing).
    notes: { type: 'string', default: 'assets/emerald-pixie-notes.txt' },
    model: { type: 'string', default: process.env.OPENROUTER_MODEL || 'google/gemini-3.1-flash-lite-image' },
    prompt: { type: 'string', default: '' },
    concurrency: { type: 'string', default: '4' },
    only: { type: 'string' },
    force: { type: 'boolean', default: false },
    'no-anchor': { type: 'boolean', default: false },
    'dry-run': { type: 'boolean', default: false },
  },
});

const notes = await fs.readFile(opt.notes, 'utf8').then(t => t.trim(), () => '');
const API = (process.env.OPENROUTER_BASE_URL || 'https://openrouter.ai/api/v1').replace(/\/$/, '');
const KEY = process.env.OPENROUTER_API_KEY;

const STYLE_RULES = `
Art style (enforce strictly, copy it from the reference image):
- clean modern anime / VTuber character-sheet illustration, crisp thin dark lineart
- soft cel shading with 2 tones per material, gentle gradient highlights in the hair
- large glossy anime eyes with highlights, small nose, soft blush
- flat solid pure white background, no scenery, no text, no watermark, no border
- full body, entire character visible head to toe, same framing and scale as the RENDER image
`.trim();

// What the camera sees at each yaw. Image models drift toward the front-facing reference, so
// spell out which features must and must not be visible.
function describeView({ yaw, pitch }) {
  const y = ((yaw % 360) + 360) % 360;
  const side = s => `her ${s} side`;
  const [facing, visible] =
    y < 22.5 || y >= 337.5 ? ['FRONT view, she faces the viewer', 'face, both eyes, chest bow and front buttons are visible']
    : y < 67.5 ? [`THREE-QUARTER FRONT view, body turned so we see the front and ${side('left')}`, 'face is seen at an angle, the far eye is partly hidden']
    : y < 112.5 ? [`exact SIDE PROFILE of ${side('left')}, she faces toward the LEFT edge of the image`, 'only ONE eye visible, nose and chin in silhouette, torso seen edge-on, the chest bow sticks out in profile, one arm in front of the body']
    : y < 157.5 ? [`THREE-QUARTER BACK view, we see her back and ${side('left')}`, 'face mostly hidden (at most a sliver of cheek), the chest bow and front buttons are NOT visible, mostly back of the hair and back of the dress']
    : y < 202.5 ? ['BACK view, she faces directly away from the viewer', 'NO face, NO eyes, NO chest bow, NO front buttons; show the back of the head, the back of the dress and the backs of the legs and shoes']
    : y < 247.5 ? [`THREE-QUARTER BACK view, we see her back and ${side('right')}`, 'face mostly hidden (at most a sliver of cheek), the chest bow and front buttons are NOT visible, mostly back of the hair and back of the dress']
    : y < 292.5 ? [`exact SIDE PROFILE of ${side('right')}, she faces toward the RIGHT edge of the image`, 'only ONE eye visible, nose and chin in silhouette, torso seen edge-on, the chest bow sticks out in profile, one arm in front of the body']
    : [`THREE-QUARTER FRONT view, body turned so we see the front and ${side('right')}`, 'face is seen at an angle, the far eye is partly hidden'];
  const cam = pitch > 5 ? `camera slightly above, looking down ${pitch}°`
    : pitch < -5 ? `camera slightly below, looking up ${-pitch}°` : 'camera at eye level';
  return { facing: `${facing}; ${cam}`, visible };
}

function buildPrompt(view, hasAnchor) {
  const { facing, visible } = describeView(view);
  return [
    `Redraw the 3D model RENDER as a polished 2D anime illustration.`,
    `VIEWING ANGLE (most important rule): ${facing}. ${visible}.`,
    `The RENDER image is the ground truth for the angle, pose and silhouette. Match it exactly. Never turn the character toward the viewer just because the reference images face forward.`,
    opt['style-only']
      ? `The STYLE REFERENCE is a different character: copy ONLY its art style, NOT its character, outfit or colors.`
      : `The CHARACTER REFERENCE is official front-view art of this exact character: copy its art style and design (face, hair, accessories, outfit details, exact colors). Where the render lacks or simplifies a detail the reference shows, add it as it would look from this angle.`,
    hasAnchor ? `The FRONT ANCHOR is an already-finished front view in the target style: keep identity, colors and details consistent with it, but do NOT copy its angle.` : '',
    `Remove all 3D-render artifacts (plastic shading, specular glare, blurry or low-poly textures); interpret them as clean fabric, hair and skin.`,
    notes ? `Character details that must appear correctly at EVERY angle (when visible from this side):\n${notes}` : '',
    STYLE_RULES,
    opt.prompt,
  ].filter(Boolean).join('\n');
}

const mimeOf = f => ({ '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp' })[path.extname(f).toLowerCase()];
const dataUrl = async f => `data:${mimeOf(f)};base64,${(await fs.readFile(f)).toString('base64')}`;

// images: [{ label, url }] — each image is preceded by its label so the model knows which is which.
async function generate(view, images) {
  const body = {
    model: opt.model,
    modalities: ['image', 'text'],
    // OpenRouter reserves credit for the worst case up front; one image is ~1.3k tokens, so cap it.
    max_tokens: 4096,
    messages: [{
      role: 'user',
      content: [
        { type: 'text', text: buildPrompt(view, images.some(i => i.label.startsWith('FRONT ANCHOR'))) },
        ...images.flatMap(({ label, url }) => [
          { type: 'text', text: `${label}:` },
          { type: 'image_url', image_url: { url } },
        ]),
        { type: 'text', text: `Now draw the illustration. Remember: ${describeView(view).facing}.` },
      ],
    }],
  };
  for (let attempt = 1; ; attempt++) {
    const headers = { 'content-type': 'application/json', 'x-title': 'makeanimereal' };
    if (KEY) headers.authorization = `Bearer ${KEY}`;
    const res = await fetch(`${API}/chat/completions`, { method: 'POST', headers, body: JSON.stringify(body) });
    const text = await res.text();
    let json;
    try { json = JSON.parse(text); } catch { json = null; }
    const img = json?.choices?.[0]?.message?.images?.[0]?.image_url?.url;
    if (res.ok && img) return img;
    const err = json?.error?.message || text.slice(0, 300);
    // Auth and real out-of-credit errors won't fix themselves; everything else (429, 5xx, empty
    // image, a 402 that only means credit is reserved by in-flight requests) gets retried.
    const fatal = [400, 401, 403].includes(res.status) || (res.status === 402 && !/in-flight/i.test(err));
    if (fatal || attempt >= 4) {
      throw new Error(`OpenRouter ${res.status}: ${err || 'no image in response'}`);
    }
    const wait = 2 ** attempt * 1000;
    console.warn(`  retry ${view.file} in ${wait / 1000}s (${res.status}: ${err || 'no image returned'})`);
    await new Promise(r => setTimeout(r, wait));
  }
}

async function save(url, file) {
  const m = /^data:image\/(\w+);base64,(.*)$/s.exec(url);
  const buf = m ? Buffer.from(m[2], 'base64') : Buffer.from(await (await fetch(url)).arrayBuffer());
  await fs.writeFile(file, buf);
}

const rendersDir = path.resolve(opt.renders);
const outDir = path.resolve(opt.out);
const manifest = JSON.parse(await fs.readFile(path.join(rendersDir, 'manifest.json'), 'utf8'));
let views = manifest.views;
if (opt.only) {
  const only = new Set(opt.only.split(','));
  views = views.filter(v => only.has(v.file));
}
await fs.mkdir(outDir, { recursive: true });

const styleUrl = await dataUrl(path.resolve(opt.style));
const exists = f => fs.access(f).then(() => true, () => false);
const done = [];
const failed = [];

async function run(view, anchorUrl) {
  const outFile = path.join(outDir, view.file);
  if (!opt.force && await exists(outFile)) return done.push(view);
  if (opt['dry-run']) {
    console.log(`--- ${view.file}\n${buildPrompt(view, !!anchorUrl)}\n`);
    return;
  }
  const images = [
    { label: `RENDER (the angle and pose to draw)`, url: await dataUrl(path.join(rendersDir, view.file)) },
    { label: opt['style-only'] ? 'STYLE REFERENCE' : 'CHARACTER REFERENCE (front view, for design and style only)', url: styleUrl },
  ];
  if (anchorUrl) images.push({ label: 'FRONT ANCHOR (for identity only, not the angle)', url: anchorUrl });
  const t = Date.now();
  try {
    await save(await generate(view, images), outFile);
    done.push(view);
    console.log(`ok   ${view.file} (${((Date.now() - t) / 1000).toFixed(1)}s)`);
  } catch (e) {
    failed.push({ file: view.file, error: e.message });
    console.error(`fail ${view.file}: ${e.message}`);
    if (/^OpenRouter 40[123]/.test(e.message)) throw e; // auth/credit problems: stop everything
  }
}

// Front, eye-level view of the first frame is generated first and becomes the identity anchor.
let anchorUrl = null;
const anchorView = manifest.views.find(v => v.frame === 0 && v.yaw === 0 && v.pitch === 0) || manifest.views[0];
const queue = views.filter(v => opt['no-anchor'] || v.file !== anchorView.file);
try {
  if (!opt['no-anchor'] && !opt['dry-run']) {
    await run(anchorView, null);
    const anchorFile = path.join(outDir, anchorView.file);
    if (await exists(anchorFile)) anchorUrl = await dataUrl(anchorFile);
  }
  await Promise.all(Array.from({ length: Math.max(1, Number(opt.concurrency)) }, async () => {
    while (queue.length) await run(queue.shift(), anchorUrl);
  }));
} catch (e) {
  console.error(`stopping: ${e.message}`);
  process.exitCode = 1;
}

if (!opt['dry-run']) {
  // List every view that has a drawing on disk, including ones from earlier or --only runs.
  const doneFiles = new Set((await fs.readdir(outDir)).filter(f => f.endsWith('.png')));
  await fs.writeFile(path.join(outDir, 'manifest.json'), JSON.stringify({
    ...manifest, model: opt.model, style: opt.style, renders: path.relative(outDir, rendersDir),
    views: manifest.views.filter(v => doneFiles.has(v.file)),
    failed,
  }, null, 2));
  console.log(`${done.length} stylized, ${failed.length} failed -> ${path.relative(process.cwd(), outDir)}`);
  if (failed.length) process.exitCode = 1;
}
