@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= NCELLS) {
    return;
  }
  let kx = wn(idx % N);
  let ky = wn(idx / N);
  if (kx == 0.0 && ky == 0.0) {
    dst[idx] = vec2<f32>(0.0, 0.0);
    return;
  }
  dst[idx] = src[idx] * inBand(kx, ky);
}
