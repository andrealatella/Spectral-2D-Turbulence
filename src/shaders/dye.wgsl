const M: u32 = $M$u;
const MF: f32 = $MF$;

struct DyeCfg {
  fade: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
}

@group(0) @binding(0) var<storage, read> vel: array<vec2<f32>>;
@group(0) @binding(1) var srcTex: texture_2d<f32>;
@group(0) @binding(2) var auxTex: texture_2d<f32>;
@group(0) @binding(3) var dstTex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var samp: sampler;
@group(0) @binding(5) var<storage, read> dtbuf: array<f32>;
@group(0) @binding(6) var<uniform> S: Splats;
@group(0) @binding(7) var<uniform> D: DyeCfg;

fn velAt(ix: i32, iy: i32) -> vec2<f32> {
  let n = i32(N);
  let x = u32(((ix % n) + n) % n);
  let y = u32(((iy % n) + n) % n);
  return vel[y * N + x];
}

fn sampleVel(p: vec2<f32>) -> vec2<f32> {
  let g = p * NF - vec2<f32>(0.5, 0.5);
  let b = floor(g);
  let f = g - b;
  let ix = i32(b.x);
  let iy = i32(b.y);
  let v00 = velAt(ix, iy);
  let v10 = velAt(ix + 1, iy);
  let v01 = velAt(ix, iy + 1);
  let v11 = velAt(ix + 1, iy + 1);
  return mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y);
}

fn trace(p: vec2<f32>, h: f32) -> vec2<f32> {
  let v1 = sampleVel(p) * h;
  let mid = fract(p - 0.5 * v1 + vec2<f32>(1.0, 1.0));
  let v2 = sampleVel(mid) * h;
  return fract(p - v2 + vec2<f32>(1.0, 1.0));
}

fn loadSrc(ix: i32, iy: i32) -> vec4<f32> {
  let m = i32(M);
  let x = ((ix % m) + m) % m;
  let y = ((iy % m) + m) % m;
  return textureLoad(srcTex, vec2<i32>(x, y), 0);
}

fn cellPos(gid: vec3<u32>) -> vec2<f32> {
  return (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5, 0.5)) / MF;
}

@compute @workgroup_size(8, 8)
fn advect_fwd(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= M || gid.y >= M) {
    return;
  }
  let s = trace(cellPos(gid), dtbuf[0] / TWO_PI);
  textureStore(dstTex, vec2<i32>(i32(gid.x), i32(gid.y)), textureSampleLevel(srcTex, samp, s, 0.0));
}

@compute @workgroup_size(8, 8)
fn advect_back(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= M || gid.y >= M) {
    return;
  }
  let s = trace(cellPos(gid), -dtbuf[0] / TWO_PI);
  textureStore(dstTex, vec2<i32>(i32(gid.x), i32(gid.y)), textureSampleLevel(srcTex, samp, s, 0.0));
}

@compute @workgroup_size(8, 8)
fn advect_corr(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= M || gid.y >= M) {
    return;
  }
  let p = cellPos(gid);
  let s = trace(p, dtbuf[0] / TWO_PI);

  let a = textureSampleLevel(srcTex, samp, s, 0.0);
  let b = textureSampleLevel(auxTex, samp, s, 0.0);
  let corrected = 1.5 * a - 0.5 * b;

  let g = s * MF - vec2<f32>(0.5, 0.5);
  let base = floor(g);
  let ix = i32(base.x);
  let iy = i32(base.y);
  let c00 = loadSrc(ix, iy);
  let c10 = loadSrc(ix + 1, iy);
  let c01 = loadSrc(ix, iy + 1);
  let c11 = loadSrc(ix + 1, iy + 1);
  let lo = min(min(c00, c10), min(c01, c11));
  let hi = max(max(c00, c10), max(c01, c11));

  var res = clamp(corrected, lo, hi) * D.fade;

  for (var i = 0u; i < S.count; i = i + 1u) {
    let sp = S.items[i];
    let rx = sep(p.x, sp.pos.x);
    let ry = sep(p.y, sp.pos.y);
    let sig2 = max(sp.radius * sp.radius, 1e-8);
    let r2 = rx * rx + ry * ry;
    if (r2 < 9.0 * sig2) {
      res = res + vec4<f32>(sp.color, 0.0) * (sp.dyeAmp * exp(-r2 / sig2));
    }
  }

  textureStore(dstTex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(res.rgb, 1.0));
}
