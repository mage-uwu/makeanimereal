#!/usr/bin/env node
// Turn every render from scripts/render.mjs into an anime illustration of the same view via an
// OpenRouter image model. The style reference image fixes the art style; the first stylized
// front view is fed back as an identity anchor so every angle draws the same character.
//
//   OPENROUTER_API_KEY=... node scripts/stylize.mjs [--renders out/renders] [--out out/stylized]
//        [--style assets/style-ref.jpg] [--model google/gemini-3.1-flash-image]
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
    model: { type: 'string', default: process.env.OPENROUTER_MODEL || 'google/gemini-3.1-flash-image' },
    prompt: { type: 'string', default: '' },
    concurrency: { type: 'string', default: '4' },
    only: { type: 'string' },
    force: { type: 'boolean', default: false },
    'no-anchor': { type: 'boolean', default: false },
    'dry-run': { type: 'boolean', default: false },
  },
});

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

function describeView({ yaw, pitch }) {
  const y = ((yaw % 360) + 360) % 360;
  const facing =
    y < 22.5 || y >= 337.5 ? 'facing the viewer (front view)'
    : y < 67.5 ? 'three-quarter view, turned toward her left (we see her left side)'
    : y < 112.5 ? 'side profile, we see her left side'
    : y < 157.5 ? 'three-quarter back view, we see her back and left side'
    : y < 202.5 ? 'back view, facing away from the viewer, face NOT visible'
    : y < 247.5 ? 'three-quarter back view, we see her back and right side'
    : y < 292.5 ? 'side profile, we see her right side'
    : 'three-quarter view, turned toward her right (we see her right side)';
  const cam = pitch > 5 ? `camera slightly above, looking down ${pitch}°`
    : pitch < -5 ? `camera slightly below, looking up ${-pitch}°` : 'camera at eye level';
  return `${facing}; ${cam}`;
}

function buildPrompt(view, hasAnchor) {
  return [
    `You are redrawing a 3D model render as a polished 2D anime illustration.`,
    opt['style-only']
      ? `Images provided: (1) STYLE REFERENCE — copy ONLY its art style, NOT its character, outfit or colors.`
      : `Images provided: (1) CHARACTER REFERENCE — the official front-view art of this exact character. Copy its art style AND its character design: face, hair, accessories, outfit details and colors.`,
    `(2) RENDER — the 3D model from a specific camera angle. Reproduce this exact pose, viewing angle,`,
    opt['style-only']
      ? `silhouette, proportions, outfit design and color palette, just drawn in the reference art style.`
      : `silhouette and proportions. Where the render lacks or simplifies a detail that the reference shows, follow the reference; work out how it looks from this angle.`,
    hasAnchor ? `(3) CHARACTER ANCHOR — an already-finished front view of this same character in the target style; keep hair, face, outfit details and colors consistent with it.` : '',
    `Camera: ${describeView(view)}. Do not rotate the character to face the viewer if the render does not.`,
    `Remove all 3D-render artifacts (plastic shading, specular glare, blurry or low-poly textures); interpret them as clean fabric, hair and skin.`,
    STYLE_RULES,
    opt.prompt,
  ].filter(Boolean).join('\n');
}

const mimeOf = f => ({ '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp' })[path.extname(f).toLowerCase()];
const dataUrl = async f => `data:${mimeOf(f)};base64,${(await fs.readFile(f)).toString('base64')}`;

async function generate(view, images) {
  const body = {
    model: opt.model,
    modalities: ['image', 'text'],
    messages: [{
      role: 'user',
      content: [
        { type: 'text', text: buildPrompt(view, images.length > 2) },
        ...images.map(url => ({ type: 'image_url', image_url: { url } })),
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
    // 401/402/403 won't fix themselves; everything else (429, 5xx, empty image) gets retried.
    if ([400, 401, 402, 403].includes(res.status) || attempt >= 4) {
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
  const images = [styleUrl, await dataUrl(path.join(rendersDir, view.file))];
  if (anchorUrl) images.push(anchorUrl);
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
  process.exit(1);
}

if (!opt['dry-run']) {
  const doneFiles = new Set(done.map(v => v.file));
  await fs.writeFile(path.join(outDir, 'manifest.json'), JSON.stringify({
    ...manifest, model: opt.model, style: opt.style, renders: path.relative(outDir, rendersDir),
    views: manifest.views.filter(v => doneFiles.has(v.file)),
    failed,
  }, null, 2));
  console.log(`${done.length} stylized, ${failed.length} failed -> ${path.relative(process.cwd(), outDir)}`);
  if (failed.length) process.exitCode = 1;
}
