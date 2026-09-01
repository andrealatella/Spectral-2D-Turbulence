@group(0) @binding(0) var<storage, read_write> force: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> S: Splats;
@group(0) @binding(2) var<uniform> P: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= NCELLS) {
    return;
  }
  let x = (f32(idx % N) + 0.5) / NF;
  let y = (f32(idx / N) + 0.5) / NF;

  var total = 0.0;
  for (var i = 0u; i < S.count; i = i + 1u) {
    let s = S.items[i];
    let rx = sep(x, s.pos.x);
    let ry = sep(y, s.pos.y);
    let r2 = rx * rx + ry * ry;
    let sig2 = max(s.radius * s.radius, 1e-8);
    if (r2 < 9.0 * sig2) {
      let g = exp(-r2 / sig2);
      total = total + 2.0 * s.amp * g * (s.vel.x * ry - s.vel.y * rx) / sig2;
    }
  }
  force[idx] = vec2<f32>(total, 0.0);
}
