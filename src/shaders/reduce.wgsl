const GROUPS: u32 = 256u;
const TH: u32 = 256u;

@group(0) @binding(0) var<storage, read> vel: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> partials: array<f32>;
@group(0) @binding(2) var<storage, read_write> dtbuf: array<f32>;
@group(0) @binding(3) var<uniform> P: Params;

var<workgroup> sdata: array<f32, TH>;

fn reduceShared(tid: u32) {
  var s = TH / 2u;
  while (s > 0u) {
    if (tid < s) {
      sdata[tid] = max(sdata[tid], sdata[tid + s]);
    }
    workgroupBarrier();
    s = s / 2u;
  }
}

@compute @workgroup_size(TH)
fn reduce_speed(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  var m = 0.0;
  var i = wid.x * TH + lid.x;
  let stride = GROUPS * TH;
  while (i < NCELLS) {
    let v = vel[i];
    m = max(m, abs(v.x) + abs(v.y));
    i = i + stride;
  }
  sdata[lid.x] = m;
  workgroupBarrier();
  reduceShared(lid.x);
  if (lid.x == 0u) {
    partials[wid.x] = sdata[0];
  }
}

@compute @workgroup_size(TH)
fn reduce_dt(@builtin(local_invocation_id) lid: vec3<u32>) {
  var m = 0.0;
  var i = lid.x;
  while (i < GROUPS) {
    m = max(m, partials[i]);
    i = i + TH;
  }
  sdata[lid.x] = m;
  workgroupBarrier();
  reduceShared(lid.x);
  if (lid.x == 0u) {
    let speed = sdata[0];
    var dt = P.dtMax;
    if (speed > 1e-8) {
      dt = min(P.dtMax, P.cfl * P.dx / speed);
    }
    dtbuf[0] = max(dt, P.dtMin);
    dtbuf[1] = speed;
  }
}
