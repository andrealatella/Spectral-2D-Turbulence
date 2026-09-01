struct RenderCfg {
  mode: u32,
  pad0: u32,
  exposure: f32,
  vortScale: f32,
  dyeGain: f32,
  pad1: f32,
  pad2: f32,
  pad3: f32,
}

@group(0) @binding(0) var dyeTex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<storage, read> phys: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> vel: array<vec2<f32>>;
@group(0) @binding(4) var<uniform> C: RenderCfg;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  var corners = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0),
  );
  let xy = corners[vi];
  var out: VSOut;
  out.pos = vec4<f32>(xy, 0.0, 1.0);
  out.uv = (xy + vec2<f32>(1.0, 1.0)) * 0.5;
  return out;
}

fn cellAt(ix: i32, iy: i32, which: u32) -> vec2<f32> {
  let n = i32(N);
  let x = u32(((ix % n) + n) % n);
  let y = u32(((iy % n) + n) % n);
  if (which == 0u) {
    return phys[y * N + x];
  }
  return vel[y * N + x];
}

fn bilinear(uv: vec2<f32>, which: u32) -> vec2<f32> {
  let g = uv * NF - vec2<f32>(0.5, 0.5);
  let b = floor(g);
  let f = g - b;
  let ix = i32(b.x);
  let iy = i32(b.y);
  let a00 = cellAt(ix, iy, which);
  let a10 = cellAt(ix + 1, iy, which);
  let a01 = cellAt(ix, iy + 1, which);
  let a11 = cellAt(ix + 1, iy + 1, which);
  return mix(mix(a00, a10, f.x), mix(a01, a11, f.x), f.y);
}

fn vortColor(t: f32) -> vec3<f32> {
  let a = clamp(t, -1.0, 1.0);
  let m = abs(a);
  let cool = vec3<f32>(0.16, 0.52, 1.00);
  let warm = vec3<f32>(1.00, 0.40, 0.12);
  var hue = cool;
  if (a > 0.0) {
    hue = warm;
  }
  let base = mix(vec3<f32>(0.035, 0.042, 0.062), hue, smoothstep(0.0, 0.60, m));
  return mix(base, vec3<f32>(1.0, 0.95, 0.88), smoothstep(0.70, 1.0, m) * 0.7);
}

fn heat(t: f32) -> vec3<f32> {
  let x = clamp(t, 0.0, 1.0);
  let r = clamp(1.55 * x - 0.18, 0.0, 1.0);
  let g = clamp(1.32 * x * x - 0.06, 0.0, 1.0);
  let b = clamp(2.20 * x * x * x * (1.0 - 0.55 * x) + 0.14 * x, 0.0, 1.0);
  return vec3<f32>(r, g, b) * 0.94 + vec3<f32>(0.03, 0.035, 0.055);
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  let uv = in.uv;
  let dye = textureSampleLevel(dyeTex, samp, uv, 0.0).rgb * C.dyeGain;
  let omega = bilinear(uv, 0u).x;
  let t = clamp(omega * C.vortScale, -1.0, 1.0);

  var col: vec3<f32>;
  if (C.mode == 0u) {
    col = dye;
  } else if (C.mode == 1u) {
    col = vortColor(t);
  } else if (C.mode == 2u) {
    let shade = vortColor(t);
    col = shade * (0.30 + 0.70 * C.exposure) + dye * (0.55 + 0.45 * abs(t));
  } else {
    let v = bilinear(uv, 1u);
    col = heat(length(v) * C.vortScale * 0.75);
  }

  col = vec3<f32>(1.0, 1.0, 1.0) - exp(-col * C.exposure);
  return vec4<f32>(pow(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(0.4545)), 1.0);
}
