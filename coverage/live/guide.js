// The guide for the edge-aware upscale (nts-gl.js's display pass): her geometry at the canvas's own
// resolution, as the G-buffer tiles hold it at the network's. One float target, 4x multisampled:
//   r, g: the view-space normal's x and y (z is the positive rest: the tiles turn back faces round)
//   b:    depth in the depth tile's units (1 at radius - 0.8, 0 at radius + 0.8), unquantised
//   a:    coverage (1 on her, 0 off; the samples' average at her edges)
// r, g, b are premultiplied by coverage, so a pixel half on her averages only the half that is.
// The camera is the network frame's, shifted and scaled so the canvas sees what the frame shows there.
export function createGuide({ THREE, renderer, vertexShader, depthUniforms }) {
  const mat = new THREE.ShaderMaterial({
    vertexShader, side: THREE.DoubleSide, uniforms: { near: depthUniforms.near, far: depthUniforms.far },
    fragmentShader: `uniform float near, far; varying vec3 vNormal; varying vec2 vUv0; varying float vDepth;
      void main() {
        vec3 n = normalize(vNormal) * (gl_FrontFacing ? 1.0 : -1.0);
        gl_FragColor = vec4(n.xy, 1.0 - clamp((vDepth - near) / (far - near), 0.0, 1.0), 1.0);
      }`,
  });
  let rt = null;
  const clear = new THREE.Color();
  // hide: objects the tiles' geometry passes leave out (they're shown for colour only)
  function render(scene, camera, { W, H, box }, hide = []) {
    if (!rt || rt.width !== W || rt.height !== H) {
      rt?.dispose();
      rt = new THREE.WebGLRenderTarget(W, H, { samples: 4, type: THREE.HalfFloatType, depthBuffer: true });
    }
    const aspect = camera.aspect, bg = scene.background, alpha = renderer.getClearAlpha(), vis = hide.map(o => o.visible);
    renderer.getClearColor(clear);
    camera.setViewOffset(box[2], box[3], -box[0], -box[1], W, H);
    scene.background = null; scene.overrideMaterial = mat; hide.forEach(o => { o.visible = false; });
    renderer.setRenderTarget(rt); renderer.setClearColor(0x000000, 0); renderer.clear(true, true, false);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null); renderer.setClearColor(clear, alpha);
    camera.clearViewOffset(); camera.aspect = aspect; camera.updateProjectionMatrix();
    scene.background = bg; scene.overrideMaterial = null; hide.forEach((o, i) => { o.visible = vis[i]; });
    return renderer.properties.get(rt.texture).__webglTexture;
  }
  return { render };
}
