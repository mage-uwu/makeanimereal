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
function run({ input, rgba, w, h, r0, r1 }) {
  const n = w * h, m = (r1 - r0) * w * 4;
  if (n > cap) {   // the allocator only grows: leave room so this happens rarely
    cap = Math.ceil(n * 1.3);
    buf = { in: k.nts_alloc(cap * 72), xa: k.nts_alloc(cap * 256), xb: k.nts_alloc(cap * 256), dist: k.nts_alloc(cap), out: k.nts_alloc(cap * 4) };
  }
  new Float32Array(k.memory.buffer, buf.in, n * 18).set(new Float32Array(input, 0, n * 18));
  const t0 = performance.now();
  k.nts_run(buf.in, w, h, r0, r1, buf.xa, buf.xb, buf.dist, buf.out);
  const ms = performance.now() - t0;
  // the caller's buffers go back to it, the output in its own (grown if it was too small)
  if (rgba.byteLength < m) rgba = new ArrayBuffer(Math.ceil(m * 1.3));
  new Uint8Array(rgba, 0, m).set(new Uint8Array(k.memory.buffer, buf.out, m));
  postMessage({ rgba, input, ms }, [rgba, input]);
}
