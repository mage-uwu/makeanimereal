// Toon contours over the neural drawing: an inverted-hull outline, as anime games draw them, at the
// canvas's own resolution (crisp at any zoom, unlike lines learned at the network's).
//   1. depth prepass: every opaque material, both sides, alpha cut out as the model cuts it
//   2. the hull: back faces pushed out along their normals by a fixed width in canvas pixels (and a
//      few millimetres back, so a hull never covers its own surface), drawn black against that depth
// What shows of the hull is her silhouette and the edges of whatever is in front of something else
// (an arm over her body). Edges in front of something close behind (her jaw over her neck, bangs over
// her forehead) the hull's few millimetres can miss, so the prepass depth goes along too: nts-gl.js
// inks a pixel when the surface a line-width away is a step behind it. Only materials that carry an outline in the model (her
// skin, clothes and shoes) and her hair get a hull; decals (eyes, brows, lashes) and the mouth don't.
// The result is a canvas-sized mask; nts-gl.js lays it over the drawing in its display pass.
export function createToon({ THREE, renderer, root, extra = [] }) {
  const HIDDEN = new THREE.MeshBasicMaterial({ visible: false });
  const uni = { width: { value: 2 }, res: { value: new THREE.Vector2(1, 1) }, back: { value: 0.004 } };
  const cache = new Map();
  const mapOf = m => m.map ?? m.uniforms?.map?.value ?? null;
  const hullable = m => m.isOutline !== true && !m.transparent && ((m.outlineWidthMode && m.outlineWidthMode !== 'none') || /HAIR/i.test(m.name) || m.isMeshBasicMaterial);
  function derived(m) {
    if (cache.has(m)) return cache.get(m);
    let pair = { depth: HIDDEN, hull: HIDDEN };
    if (m.isOutline !== true && !m.transparent && m.visible !== false) {
      const map = m.alphaTest > 0 ? mapOf(m) : null, cut = map ? { map, alphaTest: m.alphaTest } : {};
      const depth = new THREE.MeshBasicMaterial({ colorWrite: false, side: THREE.DoubleSide, ...cut });
      let hull = HIDDEN;
      if (hullable(m)) {
        // Lambert: always has the vertex normal (Basic only with skinning or an envmap). Black, opaque.
        hull = new THREE.MeshLambertMaterial({ color: 0x000000, side: THREE.BackSide, depthWrite: false, ...cut });
        hull.onBeforeCompile = sh => {
          Object.assign(sh.uniforms, { toonWidth: uni.width, toonRes: uni.res, toonBack: uni.back });
          sh.vertexShader = 'uniform float toonWidth, toonBack; uniform vec2 toonRes;\n' + sh.vertexShader.replace('#include <project_vertex>', `#include <project_vertex>
  {
    vec3 vn = normalize(transformedNormal);
    #ifdef FLIP_SIDED
      vn = -vn;   // three.js turns back faces' normals around; the hull wants the surface's own
    #endif
    vec4 cp = projectionMatrix * (mvPosition + vec4(0.0, 0.0, -toonBack, 0.0));
    vec2 dir = (projectionMatrix * vec4(vn, 0.0)).xy * toonRes;
    float len = length(dir);
    gl_Position = cp;
    if (len > 1e-6) gl_Position.xy += dir / len * (2.0 * toonWidth) / toonRes * cp.w;
  }`);
        };
        hull.customProgramCacheKey = () => 'toon-hull';
      }
      pair = { depth, hull };
    }
    cache.set(m, pair);
    return pair;
  }
  const meshes = [];
  for (const r of [root, ...extra]) r.traverse(o => { if (o.isMesh) meshes.push(o); });
  const swap = which => {
    const saved = meshes.map(o => o.material);
    for (const o of meshes) o.material = Array.isArray(o.material) ? o.material.map(m => derived(m)[which]) : derived(o.material)[which];
    return () => meshes.forEach((o, i) => { o.material = saved[i]; });
  };

  let rt = null;
  const clear = new THREE.Color();
  // view: { W, H, box: [x, y, w, h] } as nts-gl.js gets it (the network frame's place on the canvas);
  // width: line width in canvas pixels. Returns the mask (alpha: ink) and depth textures, canvas-sized,
  // with what nts-gl.js needs to read the depth.
  function render(scene, camera, { W, H, box }, width) {
    if (!rt || rt.width !== W || rt.height !== H) {
      rt?.dispose();
      rt = new THREE.WebGLRenderTarget(W, H, { samples: 4, depthBuffer: true, depthTexture: new THREE.DepthTexture(W, H, THREE.FloatType) });
    }
    const aspect = camera.aspect, bg = scene.background, alpha = renderer.getClearAlpha(), vis = extra.map(o => o.visible);
    renderer.getClearColor(clear);
    // the frame's camera, shifted and scaled so the canvas sees exactly what the frame shows there
    camera.setViewOffset(box[2], box[3], -box[0], -box[1], W, H);
    uni.width.value = width; uni.res.value.set(W, H);
    scene.background = null; extra.forEach(o => { o.visible = true; });
    renderer.setRenderTarget(rt); renderer.setClearColor(0x000000, 0); renderer.clear(true, true, false);
    renderer.autoClear = false;
    for (const which of ['depth', 'hull']) { const restore = swap(which); renderer.render(scene, camera); restore(); }
    renderer.autoClear = true;
    renderer.setRenderTarget(null); renderer.setClearColor(clear, alpha);
    camera.clearViewOffset(); camera.aspect = aspect; camera.updateProjectionMatrix();
    scene.background = bg; extra.forEach((o, i) => { o.visible = vis[i]; });
    const tex = t => renderer.properties.get(t).__webglTexture;
    return { ink: tex(rt.texture), depth: tex(rt.depthTexture), near: camera.near, far: camera.far, width };
  }
  return { render };
}
