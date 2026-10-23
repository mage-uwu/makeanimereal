// One CPU-mode worker (nts.js): the nanotss shader's network (infer.c nts_run) on one band of rows.
let k = null, cap = 0, buf = null;
onmessage = async ({ data: m }) => {
  if (m.init) {
    k = (await WebAssembly.instantiate(m.module)).exports;
    const p = k.nts_alloc(m.net.byteLength);
    new Uint8Array(k.memory.buffer).set(new Uint8Array(m.net), p);
    if (!k.nts_load(p, m.net.byteLength)) throw new Error('net.bin did not load');
    postMessage({ ready: true });
    return;
  }
  try { run(m); } catch (e) { postMessage({ error: String(e?.stack || e) }); }
};
function run({ input, w, h, r0, r1 }) {
  const n = w * h;
  if (n > cap) {   // the allocator only grows: leave room so this happens rarely
    cap = Math.ceil(n * 1.3);
    buf = { in: k.nts_alloc(cap * 72), xa: k.nts_alloc(cap * 256), xb: k.nts_alloc(cap * 256), dist: k.nts_alloc(cap), out: k.nts_alloc(cap * 4) };
  }
  new Float32Array(k.memory.buffer, buf.in, n * 18).set(input);
  k.nts_run(buf.in, w, h, r0, r1, buf.xa, buf.xb, buf.dist, buf.out);
  const rgba = new Uint8Array(k.memory.buffer, buf.out, (r1 - r0) * w * 4).slice();
  postMessage({ rgba: rgba.buffer }, [rgba.buffer]);
}
