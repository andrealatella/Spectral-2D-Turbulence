const NBINS: u32 = $NBINS$u;
const FIXED_BIN: f32 = 1e8;
const FIXED_TOT: f32 = 1e6;
const TH: u32 = 64u;

@group(0) @binding(0) var<storage, read> w: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> bins: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> phys: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> zonal: array<f32>;

@compute @workgroup_size(64)
fn clear_bins(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= NBINS + 2u) {
    return;
  }
  atomicStore(&bins[gid.x], 0u);
}

@compute @workgroup_size(64)
fn accumulate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= NCELLS) {
    return;
  }
  let kx = wn(idx % N);
  let ky = wn(idx / N);
  let k2 = kx * kx + ky * ky;
  if (k2 == 0.0 || inBand(kx, ky) == 0.0) {
    return;
  }
  let a = w[idx];
  let n2 = NF * NF;
  let ens = 0.5 * (a.x * a.x + a.y * a.y) / (n2 * n2);
  let ene = ens / k2;
  let b = min(u32(sqrt(k2) + 0.5), NBINS - 1u);
  atomicAdd(&bins[b], u32(ene * FIXED_BIN));
  atomicAdd(&bins[NBINS], u32(ene * FIXED_TOT));
  atomicAdd(&bins[NBINS + 1u], u32(ens * FIXED_TOT));
}

var<workgroup> rowsum: array<f32, TH>;

@compute @workgroup_size(TH)
fn zonal_mean(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = wid.x;
  var acc = 0.0;
  var i = lid.x;
  while (i < N) {
    acc = acc + phys[row * N + i].y;
    i = i + TH;
  }
  rowsum[lid.x] = acc;
  workgroupBarrier();
  var s = TH / 2u;
  while (s > 0u) {
    if (lid.x < s) {
      rowsum[lid.x] = rowsum[lid.x] + rowsum[lid.x + s];
    }
    workgroupBarrier();
    s = s / 2u;
  }
  if (lid.x == 0u) {
    zonal[row] = rowsum[0] / NF;
  }
}
