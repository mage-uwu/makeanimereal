// An anime backdrop behind her: a sky and a ground she stands on, drawn per screen pixel from the live
// camera (a ray through the pixel), so it turns, tilts and zooms with the view. Two looks:
//   1 'sky'    a pastel day: sky from blush at the horizon to clear blue, two-tone cel clouds, a meadow
//              of banded toon grass with flowers, a cream stage disc under her, haze toward the horizon
//   2 'stage'  an idol stage at night: indigo sky with twinkling stars, a dark glossy floor with ring
//              lights and a spotlight pool around her
// Both put a crisp contact shadow under her feet. Seen from below the ground (the low orbit), the
// ground fades out and she floats in the sky.
//
// BACKDROP_GLSL defines vec3 backdrop(vec2 p) (p: the network frame's [0, 1] coordinates, top-down,
// any value off the frame) in display sRGB; the GPU engine's display shader includes it.
// createBackdropCanvas() draws it on its own small WebGL2 canvas for the 2D engines (CPU, WebGPU).
export const BACKDROP_GLSL = `
uniform mat4 bgInvVP;      // NDC -> world
uniform vec3 bgCenter;     // the point under her (x, ground height, z)
uniform int bgStyle;       // 1 sky, 2 stage
uniform float bgTime;      // seconds (stars twinkle)
float bgHash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float bgNoise(vec2 p) {
  vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(bgHash(i), bgHash(i + vec2(1, 0)), f.x), mix(bgHash(i + vec2(0, 1)), bgHash(i + vec2(1, 1)), f.x), f.y);
}
float bgFbm(vec2 p) { float s = 0.0, a = 0.5; for (int i = 0; i < 4; i++) { s += a * bgNoise(p); p = p * 2.03 + 17.0; a *= 0.5; } return s; }
vec3 bgSky(vec3 rd) {
  float e = clamp(rd.y, -0.2, 1.0);
  if (bgStyle == 2) {
    vec3 c = mix(vec3(0.20, 0.14, 0.38), vec3(0.05, 0.05, 0.16), smoothstep(0.0, 0.7, e));
    c += vec3(0.45, 0.25, 0.55) * exp(-max(e, 0.0) * 9.0) * 0.5;               // the city glow at the horizon
    vec2 sp = rd.xz / max(rd.y + 0.35, 0.05) * 60.0;
    float st = bgHash(floor(sp)); float tw = 0.6 + 0.4 * sin(bgTime * 2.0 + st * 40.0);
    c += vec3(1.0, 0.95, 1.0) * step(0.985, st) * smoothstep(0.35, 0.0, length(fract(sp) - 0.5)) * tw * smoothstep(-0.05, 0.15, e);
    return c;
  }
  vec3 c = mix(vec3(1.0, 0.89, 0.92), vec3(0.55, 0.78, 0.98), smoothstep(-0.02, 0.55, e));
  // cel clouds on a layer overhead: two tones, crisp edges
  vec2 cp = rd.xz / max(rd.y + 0.12, 0.04) * 0.9 + vec2(bgTime * 0.004, 0.0);
  float d = bgFbm(cp) - 0.08 * (1.0 - smoothstep(0.0, 0.4, e));
  float body = smoothstep(0.58, 0.6, d), shade = smoothstep(0.58, 0.6, bgFbm(cp + vec2(0.06, 0.05)));
  vec3 cloud = mix(vec3(0.86, 0.85, 0.96), vec3(1.0), shade);
  return mix(c, cloud, body * smoothstep(-0.02, 0.08, e));
}
vec3 backdrop(vec2 p) {
  vec2 ndc = vec2(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0);
  vec4 a = bgInvVP * vec4(ndc, -1.0, 1.0), b = bgInvVP * vec4(ndc, 1.0, 1.0);
  vec3 ro = a.xyz / a.w, rd = normalize(b.xyz / b.w - ro);
  vec3 sky = bgSky(rd);
  float above = smoothstep(bgCenter.y - 0.02, bgCenter.y + 0.25, ro.y);      // under the ground: no ground
  if (rd.y >= -1e-4 || above <= 0.0) return sky;
  float t = (bgCenter.y - ro.y) / rd.y;
  vec3 h = ro + rd * t;
  vec2 g = h.xz - bgCenter.xz;
  float r = length(g);
  vec3 c;
  if (bgStyle == 2) {
    c = vec3(0.10, 0.08, 0.20);
    float ring = abs(fract(r * 1.2) - 0.5);
    c += vec3(0.55, 0.35, 0.9) * smoothstep(0.03, 0.0, ring) * smoothstep(6.0, 1.0, r) * 0.7;   // ring lights
    c += vec3(1.0, 0.75, 0.95) * smoothstep(1.6, 0.0, r) * 0.45;                                  // spotlight pool
    c = mix(c, sky * 0.6 + c * 0.4, 0.35 * smoothstep(0.0, 1.0, 1.0 - abs(rd.y)));                  // gloss: the sky in the floor
  } else {
    float band = step(0.5, bgNoise(h.xz * 1.3 + bgFbm(h.xz * 0.3) * 2.0));                         // two-tone toon grass
    c = mix(vec3(0.56, 0.80, 0.47), vec3(0.66, 0.87, 0.55), band);
    vec2 cell = floor(h.xz * 2.5);
    float fl = step(0.93, bgHash(cell)) * smoothstep(0.18, 0.1, length(fract(h.xz * 2.5) - 0.5));  // little flowers
    c = mix(c, mix(vec3(1.0, 0.75, 0.85), vec3(1.0, 0.98, 0.9), bgHash(cell + 3.0)), fl);
    float disc = smoothstep(1.02, 1.0, r);                                                         // the stage disc, cream with a pink rim
    c = mix(c, mix(vec3(1.0, 0.95, 0.88), vec3(0.98, 0.72, 0.80), smoothstep(0.86, 0.88, r)), disc);
  }
  // her contact shadow: crisp, a little soft at the edge
  float sh = smoothstep(0.34, 0.28, length(g * vec2(1.0, 1.25)));
  c = mix(c, c * (bgStyle == 2 ? vec3(0.35, 0.3, 0.5) : vec3(0.70, 0.66, 0.85)), sh * 0.85);
  // haze toward the horizon
  vec3 haze = bgStyle == 2 ? vec3(0.22, 0.15, 0.36) : vec3(1.0, 0.9, 0.93);
  c = mix(c, haze, smoothstep(4.0, 40.0, t));
  return mix(sky, c, above);
}
`;

// Uniform values for the camera and where she stands.
export function backdropUniforms(THREE, camera, center, style, time) {
  const m = new THREE.Matrix4().multiplyMatrices(camera.matrixWorld, camera.projectionMatrixInverse);
  return { invVP: Float32Array.from(m.elements), center: [center.x, center.y, center.z], style, time };
}
export function setBackdropUniforms(gl, u, b) {
  gl.uniformMatrix4fv(u.bgInvVP, false, b.invVP);
  gl.uniform3f(u.bgCenter, ...b.center); gl.uniform1i(u.bgStyle, b.style); gl.uniform1f(u.bgTime, b.time);
}

// For the 2D engines: the backdrop on its own canvas, the size of the view; draw it, then her over it.
export function createBackdropCanvas() {
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl2', { premultipliedAlpha: false, preserveDrawingBuffer: true });
  if (!gl) return null;
  const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
  const p = gl.createProgram();
  gl.attachShader(p, sh(gl.VERTEX_SHADER, `#version 300 es
void main() { vec2 q = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)); gl_Position = vec4(q * 2.0 - 1.0, 0.0, 1.0); }`));
  gl.attachShader(p, sh(gl.FRAGMENT_SHADER, `#version 300 es
precision highp float;
uniform vec4 box; uniform vec2 view;
${BACKDROP_GLSL}
out vec4 o;
void main() { vec2 q = vec2(gl_FragCoord.x, view.y - gl_FragCoord.y); o = vec4(backdrop((q - box.xy) / box.zw), 1.0); }`));
  gl.linkProgram(p);
  const u = {};
  for (let i = 0, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); i < n; i++) { const name = gl.getActiveUniform(p, i).name; u[name] = gl.getUniformLocation(p, name); }
  const vao = gl.createVertexArray();
  return {
    canvas,
    render(W, H, box, b) {
      if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
      gl.viewport(0, 0, W, H); gl.useProgram(p); gl.bindVertexArray(vao);
      gl.uniform4f(u.box, ...box); gl.uniform2f(u.view, W, H);
      setBackdropUniforms(gl, u, b);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      return canvas;
    },
  };
}
