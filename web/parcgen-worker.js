// The parcgen runtime (parcgen/parcgen.c as web/parcgen.wasm) in a worker, so generating a window
// never stalls rendering. Messages:
//   in  {type: 'init', wasm, weights}            ArrayBuffers: the module and the float32 weight blob
//   in  {type: 'terrain', heights, nx, ny, minX, minY, cell}
//   in  {type: 'generate', id, prev, target, stride, cfg}   prev: 2 x 78 floats (world)
//   out {type: 'ready'} | {type: 'window', id, frames (15 x 78), ms} | {type: 'error', message}
let x = null;

const gaussian = () => {
  let u = 0;
  while (u === 0) u = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
};

onmessage = async ({ data }) => {
  try {
    if (data.type === 'init') {
      const { instance } = await WebAssembly.instantiate(data.wasm, {});
      x = instance.exports;
      if (x.pg_create() !== 0) throw new Error('parcgen: out of memory');
      if (data.weights.byteLength !== x.pg_num_weights() * 4) throw new Error('parcgen: weights do not match the runtime');
      new Uint8Array(x.memory.buffer, x.pg_weights(), data.weights.byteLength).set(new Uint8Array(data.weights));
      x.pg_bind();
      postMessage({ type: 'ready' });
    } else if (data.type === 'terrain') {
      new Float32Array(x.memory.buffer, x.pg_terrain(data.nx, data.ny, data.minX, data.minY, data.cell), data.nx * data.ny).set(data.heights);
    } else if (data.type === 'generate') {
      new Float32Array(x.memory.buffer, x.pg_prev(), 156).set(data.prev);
      const noise = new Float32Array(x.memory.buffer, x.pg_noise(), 15 * 91);
      for (let i = 0; i < noise.length; i++) noise[i] = gaussian();
      const t0 = performance.now();
      x.pg_generate(data.target[0], data.target[1], data.stride, data.cfg);
      const frames = new Float32Array(x.memory.buffer, x.pg_out(), 15 * 78).slice();
      postMessage({ type: 'window', id: data.id, frames, ms: performance.now() - t0 }, [frames.buffer]);
    }
  } catch (e) {
    postMessage({ type: 'error', message: String(e.message || e) });
  }
};
