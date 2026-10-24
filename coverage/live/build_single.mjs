// cpu.html (the Tree-SSM page) as one HTML file: every module, the worker, the wasm kernels and the
// networks' small files (meta.json, net.png) inlined. Only the big binaries stay beside it, at the
// paths the page already uses: model.vrm.png, motions.json, nts/fine0..2.png, nts-tree/fine0..2.png and
// each further character's <name>/model.vrm.png and <name>/nts-tree/fine0..2.png (an artifact page is
// capped at 16 MB; those are ~35 MB a character).
//   node build_single.mjs [out.html]   (default: single/index.html)
// The modules are bundled into the page's module script; fetch() and the network's <img> loads are
// served from the inlined bytes first, so the sources run unchanged apart from the worker's URL and
// nts-gl.js's image loader, rewritten here.
import { readFileSync, writeFileSync, mkdirSync, readdirSync, existsSync } from 'node:fs';
import path from 'node:path';

const dir = path.dirname(new URL(import.meta.url).pathname);
const out = process.argv[2] || path.join(dir, 'single/index.html');
const read = f => readFileSync(path.join(dir, f));
const b64 = b => Buffer.from(b).toString('base64');
const must = (s, a, b) => { if (!s.includes(a)) throw new Error(`build_single: '${a}' not found`); return s.split(a).join(b); };

// the worker: a blob made from its text (blob: workers are allowed in an artifact)
const worker = read('nts-worker.js').toString();
// the modules, bundled into the page's own module script: each in a function scope, its exports
// returned (an artifact page loads scripts only from its allowed CDNs, so no data: modules)
const mods = ['motion.js', 'nts.js', 'backdrop.js', 'nts-gl.js', 'toon.js', 'guide.js', 'sword.js', 'gpu-nts.js'];   // dependencies first
const exported = {};
const bundle = mods.map(m => {
  let code = read(m).toString();
  if (m === 'nts.js') code = must(code, "new Worker(new URL('./nts-worker.js', import.meta.url))",
    `new Worker(URL.createObjectURL(new Blob([${JSON.stringify(worker)}], { type: 'text/javascript' })))`);
  if (m === 'nts-gl.js') code = must(code, 'img.src = url;', 'img.src = globalThis.inlineUrl?.(url) ?? url;');
  code = code.replace(/^import \{([^}]*)\} from '\.\/([\w.-]+)';$/gm, (_, names, from) => {
    if (!exported[from]) throw new Error(`build_single: ${m} imports ${from} before it is bundled`);
    return `const {${names}} = ${exported[from]};`;
  });
  const names = [...code.matchAll(/^export (?:async )?(?:function|const) (\w+)/gm)].map(x => x[1]);
  code = code.replace(/^export (async )?function/gm, '$1function').replace(/^export const /gm, 'const ');
  if (/^(export|import) /m.test(code) || /import\.meta/.test(code)) throw new Error(`build_single: ${m} has an import or export the bundler doesn't handle`);
  exported[m] = `bundled[${JSON.stringify(m)}]`;
  return `// ---- ${m} ----\nbundled[${JSON.stringify(m)}] = (() => {\n${code}\nreturn { ${names.join(', ')} };\n})();\n`;
}).join('\n');

// the inlined files, by the paths the page fetches; the wasm is the same for both networks
const files = {};
// every character's network folders (the page's CHARS): nts/, nts-tree/, and <character>/nts-tree/
const nets = ['nts/', 'nts-tree/', ...readdirSync(dir, { withFileTypes: true }).filter(d => d.isDirectory() && existsSync(path.join(dir, d.name, 'nts-tree/meta.json'))).map(d => `${d.name}/nts-tree/`)];
for (const net of nets) {
  for (const f of ['meta.json', 'net.png', 'infer.wasm', 'infer-dot.wasm', 'infer-relaxed.wasm', 'infer-relaxed-dot.wasm']) files[net + f] = read(net + f);
}
const types = { '.json': 'application/json', '.png': 'image/png', '.wasm': 'application/wasm' };
const blobs = {}, seen = new Map();
for (const [k, v] of Object.entries(files)) {
  const s = b64(v);
  blobs[k] = seen.has(s) ? { same: seen.get(s) } : { type: types[path.extname(k)], b64: s };
  if (!seen.has(s)) seen.set(s, k);
}
const shim = `<script>
// The inlined files (build_single.mjs): fetch() and inlineUrl() serve these paths from the page itself.
(() => {
  const F = ${JSON.stringify(blobs)}, cache = {};
  const base = new URL('.', location.href).pathname;   // paths relative to the page, as the page asks for them
  const key = u => { try { const x = new URL(u, location.href); return x.origin === location.origin && x.pathname.startsWith(base) ? x.pathname.slice(base.length) : null; } catch { return null; } };
  const blob = k => {
    if (!(k in F)) return null;
    const e = F[k].same ? F[F[k].same] : F[k];
    return cache[k] ||= new Blob([Uint8Array.from(atob(e.b64), c => c.charCodeAt(0))], { type: e.type });
  };
  const fetch0 = fetch.bind(globalThis);
  globalThis.fetch = (u, o) => { const b = typeof u === 'string' || u instanceof URL ? blob(key(String(u))) : null; return b ? Promise.resolve(new Response(b, { headers: { 'content-type': b.type } })) : fetch0(u, o); };
  globalThis.inlineUrl = u => { const b = blob(key(u)); return b ? URL.createObjectURL(b) : null; };
})();
</script>
`;

let html = read('cpu.html').toString();
html = html.replace(/<script type="importmap">/, () => `${shim}<script type="importmap">`);
html = html.replace(/^import \{([^}]*)\} from '\.\/([\w.-]+)';$/gm, (_, names, from) => `const {${names}} = ${exported[from]};`);
const at = html.indexOf('<script type="module">\n') + '<script type="module">\n'.length;
// the page's own imports (three, three-vrm) stay first; the bundle goes after them
const head = html.slice(at).match(/^(?:\/\/.*\n|import .*\n)*/)[0];
html = html.slice(0, at) + head + `const bundled = {};\n${bundle}\n// ---- the page ----\n` + html.slice(at + head.length);
if (/from '\.\//.test(html)) throw new Error('build_single: a relative import is left in the page');
html = html.replace('<title>', '<!-- built by build_single.mjs from cpu.html and its modules: edit those, not this -->\n<title>');
mkdirSync(path.dirname(out), { recursive: true });
writeFileSync(out, html);
console.log(`${out}: ${(html.length / 1e6).toFixed(2)} MB; beside it: model.vrm.png, motions.json, nts/fine0..2.png, nts-tree/fine0..2.png`);
