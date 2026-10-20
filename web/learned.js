// Learned render: her hidden mesh, rasterised into a G-buffer, run through a small learned shader
// (a U-Net) that was fitted to the key drawings. Imported by the viewer (web/index.html) and by the
// training-data renderer (scripts/gbuffer-page.html), so the network sees exactly the same inputs
// live as it saw in training.
//
// The G-buffer is drawn in passes of four channels each, all from the same vertex shader:
//   pass 0  base colour rgb, coverage
//   pass 1  view-space normal xyz, depth (view distance minus the orbit radius, halved)
//   pass 2  learned features 0-3, sampled from a 3D grid around her at the surface point
//   pass 3  learned features 4-7
//   pass 4  surface point in the grid's unit cube xyz, coverage (training only: the features are
//           learned there, so the trainer samples the grid itself)

export const GBUFFER = { width: 400, height: 600 };

const vertex = /* glsl */ `
  uniform vec3 uGridMin, uGridSize;
  uniform float uRadius;
  varying vec2 vUv;
  varying vec3 vNormal, vGrid;
  varying float vDepth;
  void main() {
    vUv = uv;
    vNormal = normalMatrix * normal;
    vGrid = (position - uGridMin) / uGridSize;
    vec4 view = modelViewMatrix * vec4(position, 1.0);
    vDepth = (-view.z - uRadius) * 0.5;
    gl_Position = projectionMatrix * view;
  }`;

const fragment = /* glsl */ `
  precision highp float;
  precision highp sampler3D;
  uniform int uPass;
  uniform sampler2D uAlbedo;
  uniform sampler3D uGrid0, uGrid1;
  uniform vec4 uLo0, uLo1, uSpan0, uSpan1;
  varying vec2 vUv;
  varying vec3 vNormal, vGrid;
  varying float vDepth;
  void main() {
    if (uPass == 0) gl_FragColor = vec4(texture2D(uAlbedo, vUv).rgb, 1.0);
    else if (uPass == 1) gl_FragColor = vec4(normalize(vNormal), vDepth);
    else if (uPass == 2) gl_FragColor = uLo0 + uSpan0 * texture(uGrid0, vGrid);
    else if (uPass == 3) gl_FragColor = uLo1 + uSpan1 * texture(uGrid1, vGrid);
    else gl_FragColor = vec4(vGrid, 1.0);
  }`;

// mesh: mesh.json (position, index, uv); albedo: a THREE.Texture of the base colour; box: the
// feature grid's bounds { min: [x,y,z], size: [x,y,z] }.
export function makeGBuffer(THREE, mesh, albedo, box, radius) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(mesh.position, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(mesh.uv, 2));
  g.setIndex(mesh.index);
  g.computeVertexNormals();
  albedo.flipY = false;
  albedo.colorSpace = THREE.NoColorSpace;
  albedo.needsUpdate = true;
  const material = new THREE.ShaderMaterial({
    vertexShader: vertex, fragmentShader: fragment,
    uniforms: {
      uPass: { value: 0 }, uAlbedo: { value: albedo }, uRadius: { value: radius },
      uGridMin: { value: new THREE.Vector3(...box.min) }, uGridSize: { value: new THREE.Vector3(...box.size) },
      uGrid0: { value: null }, uGrid1: { value: null },
      uLo0: { value: new THREE.Vector4() }, uLo1: { value: new THREE.Vector4() },
      uSpan0: { value: new THREE.Vector4() }, uSpan1: { value: new THREE.Vector4() },
    },
  });
  const scene = new THREE.Scene();
  const m = new THREE.Mesh(g, material);
  m.frustumCulled = false;
  scene.add(m);
  const { width, height } = GBUFFER;
  const target = new THREE.WebGLRenderTarget(width, height, { type: THREE.FloatType, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });

  return {
    // The learned grid: 8 channels quantised to bytes, two RGBA volumes, with each channel's range.
    setGrid(bytes, dims, lo, span) {
      const [nx, ny, nz] = dims, n = nx * ny * nz;
      for (let h = 0; h < 2; h++) {
        const data = new Uint8Array(n * 4);
        for (let c = 0; c < 4; c++) { const src = (h * 4 + c) * n; for (let i = 0; i < n; i++) data[i * 4 + c] = bytes[src + i]; }
        const t = new THREE.Data3DTexture(data, nx, ny, nz);
        t.format = THREE.RGBAFormat; t.type = THREE.UnsignedByteType;
        t.minFilter = t.magFilter = THREE.LinearFilter;
        t.wrapS = t.wrapT = t.wrapR = THREE.ClampToEdgeWrapping;
        t.unpackAlignment = 1; t.needsUpdate = true;
        material.uniforms[`uGrid${h}`].value = t;
        material.uniforms[`uLo${h}`].value.fromArray(lo.slice(h * 4, h * 4 + 4));
        material.uniforms[`uSpan${h}`].value.fromArray(span.slice(h * 4, h * 4 + 4));
      }
    },
    // Render the given passes from `camera`; returns one Float32Array (RGBA, bottom row first) each.
    read(renderer, camera, passes) {
      const keep = { target: renderer.getRenderTarget(), color: renderer.getClearColor(new THREE.Color()), alpha: renderer.getClearAlpha(), auto: renderer.autoClear };
      renderer.setClearColor(0x000000, 0);
      renderer.autoClear = true;
      const out = passes.map(p => {
        material.uniforms.uPass.value = p;
        renderer.setRenderTarget(target);
        renderer.render(scene, camera);
        const buf = new Float32Array(width * height * 4);
        renderer.readRenderTargetPixels(target, 0, 0, width, height, buf);
        return buf;
      });
      renderer.setRenderTarget(keep.target);
      renderer.setClearColor(keep.color, keep.alpha);
      renderer.autoClear = keep.auto;
      return out;
    },
  };
}

// The learned shader itself, in TensorFlow.js: weights from net.json + net.bin (float16), run on
// the G-buffer passes 0-3 (16 channels). Returns straight RGBA in 0..1, top row first.
export async function loadNet(tf, base) {
  const spec = await (await fetch(`${base}/net.json`)).json();
  const raw = new Uint16Array(await (await fetch(`${base}/net.bin`)).arrayBuffer());
  const half = h => {
    const s = h >> 15, e = (h >> 10) & 31, f = h & 1023;
    const v = e === 0 ? f / 1024 * 2 ** -14 : e === 31 ? (f ? NaN : Infinity) : (1 + f / 1024) * 2 ** (e - 15);
    return s ? -v : v;
  };
  const W = {};
  for (const [name, { shape, offset }] of Object.entries(spec.tensors)) {
    const n = shape.reduce((a, b) => a * b, 1), f = new Float32Array(n);
    for (let i = 0; i < n; i++) f[i] = half(raw[offset + i]);
    W[name] = tf.tensor(f, shape); // conv weights already [kh, kw, in, out]
  }
  const conv = (x, name, act = true) => {
    const y = tf.add(tf.conv2d(x, W[`${name}.w`], 1, 'same'), W[`${name}.b`]);
    return act ? tf.leakyRelu(y, 0.1) : y;
  };
  const block = (x, name) => conv(conv(x, `${name}.0`), `${name}.1`);
  const levels = spec.levels;
  const forward = x => {
    const skips = [];
    for (let i = 0; i < levels; i++) {
      x = block(x, `down${i}`);
      skips.push(x);
      x = tf.avgPool(x, 2, 2, 'valid');
    }
    x = block(x, 'mid');
    for (let i = levels - 1; i >= 0; i--) {
      const [h, w] = skips[i].shape.slice(1, 3);
      x = tf.image.resizeNearestNeighbor(x, [h, w]);
      x = block(tf.concat([x, skips[i]], 3), `up${i}`);
    }
    return tf.sigmoid(conv(x, 'out', false));
  };
  return {
    spec,
    // passes: Float32Array RGBA per G-buffer pass 0-3, bottom row first, as gbuffer.read returns.
    run(passes) {
      return tf.tidy(() => {
        const { width, height } = GBUFFER;
        const x = tf.reverse(tf.concat(passes.map(p => tf.tensor4d(p, [1, height, width, 4])), 3), 1);
        return forward(x).squeeze();
      });
    },
  };
}
