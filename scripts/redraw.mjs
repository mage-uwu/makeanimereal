#!/usr/bin/env node
// Redraw single drawings by their key ID, then rebuild the set they belong to.
//
//   OPENROUTER_API_KEY=... node scripts/redraw.mjs <id> [<id> ...] [--prompt "extra rule"] [--hide-face] [--no-build]
//
// One ID scheme for every drawing, the same one the viewer shows under the key map:
//   still/y045/p15        a still key (the idle pose), yaw 45, pitch 15
//   walk/f06/y135/p15     walk cycle frame 6 (from 0), yaw 135, pitch 15
// The drawing being replaced is kept in <drawings>/replaced/ with a timestamp, so a bad redraw can
// be put back by copying it over. Animation frames are redrawn with the same references as the
// first pass (the still key at that angle and the previous frame); the next frame is not redrawn,
// so redraw it too if the change carries over.
import { spawnSync } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import { parseArgs } from 'node:util';

export const SETS = {
  still: {
    renders: 'out/g2-renders', drawings: 'out/g2',
    build: ['python3', 'scripts/build-web.py', '--renders', 'out/g2-renders', '--alpha', 'out/g2-alpha', '--keys', 'out/g2', '--out', 'out/web2', '--cache', 'out/cache2'],
  },
  walk: {
    renders: 'out/walk-renders', drawings: 'out/walk', stills: 'out/g2',
    build: ['python3', 'scripts/build-walk.py', '--renders', 'out/walk-alpha', '--keys', 'out/walk', '--out', 'out/web2', '--name', 'walk'],
  },
};

// "walk/f06/y135/p15" -> { set, frame, yaw, pitch }
export function parseKey(id) {
  const m = /^(\w+)\/(?:f(\d+)\/)?y(-?[\d.]+)\/p(-?[\d.]+)$/.exec(id.trim());
  if (!m || !SETS[m[1]]) throw new Error(`not a key ID: ${id} (expected still/y045/p15 or walk/f06/y135/p15)`);
  return { set: m[1], frame: m[2] === undefined ? 0 : Number(m[2]), yaw: Number(m[3]), pitch: Number(m[4]) };
}

// The manifest view (and so the render and drawing file) a key ID names.
export async function viewFor(id) {
  const key = parseKey(id), set = SETS[key.set];
  const manifest = JSON.parse(await fs.readFile(path.join(set.renders, 'manifest.json'), 'utf8'));
  const view = manifest.views.find(v => v.frame === key.frame && v.yaw === key.yaw && v.pitch === key.pitch);
  if (!view) throw new Error(`${id}: no such view in ${set.renders}`);
  return { key, set, view };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const { values: opt, positionals: ids } = parseArgs({
    allowPositionals: true,
    options: { prompt: { type: 'string', default: '' }, 'hide-face': { type: 'boolean', default: false }, 'no-build': { type: 'boolean', default: false } },
  });
  if (!ids.length) { console.error('usage: node scripts/redraw.mjs <id> [<id> ...] [--prompt "..."] [--hide-face] [--no-build]'); process.exit(2); }
  const targets = await Promise.all(ids.map(viewFor));
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const rebuild = new Set();
  for (const name of new Set(targets.map(t => t.key.set))) {
    const group = targets.filter(t => t.key.set === name), set = SETS[name];
    await fs.mkdir(path.join(set.drawings, 'replaced'), { recursive: true });
    for (const { view } of group) {
      const file = path.join(set.drawings, view.file);
      await fs.copyFile(file, path.join(set.drawings, 'replaced', view.file.replace('.png', `.${stamp}.png`))).catch(() => {});
    }
    const args = ['--no-warnings', 'scripts/stylize.mjs', '--renders', set.renders, '--out', set.drawings,
      '--only', group.map(t => t.view.file).join(','), '--force', '--concurrency', '4'];
    if (set.stills) args.push('--stills', set.stills);
    if (opt.prompt) args.push('--prompt', opt.prompt);
    if (opt['hide-face']) args.push('--hide-face');
    console.log(`redrawing ${group.map(t => t.view.file).join(', ')}`);
    const r = spawnSync('node', args, { stdio: 'inherit', env: { ...process.env, NODE_USE_ENV_PROXY: '1' } });
    if (r.status) process.exit(r.status);
    rebuild.add(name);
  }
  if (!opt['no-build']) for (const name of rebuild) {
    console.log(`rebuilding ${name}`);
    const [cmd, ...args] = SETS[name].build;
    const r = spawnSync(cmd, args, { stdio: 'inherit' });
    if (r.status) process.exit(r.status);
  }
}
