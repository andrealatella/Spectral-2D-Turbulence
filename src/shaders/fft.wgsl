const N: u32 = $N$u;
const WG: u32 = $WG$u;
const AXIS: u32 = $AXIS$u;
const R4_STAGES: u32 = $R4_STAGES$u;
const SGN: f32 = $SGN$;
const SCALE: f32 = $SCALE$;
const TWO_PI: f32 = 6.28318530717958647;

@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;

var<workgroup> sh: array<vec2<f32>, N>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn twiddle(ang: f32) -> vec2<f32> {
  return vec2<f32>(cos(ang), sin(ang));
}

fn rot(z: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(-SGN * z.y, SGN * z.x);
}

fn gidx(line: u32, i: u32) -> u32 {
  if (AXIS == 0u) {
    return line * N + i;
  }
  return i * N + line;
}

fn digitReverse(idx: u32) -> u32 {
  var out = 0u;
  var v = idx;
$REV_R2$  for (var s = 0u; s < R4_STAGES; s = s + 1u) {
    out = out * 4u + (v % 4u);
    v = v / 4u;
  }
  return out;
}

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let line = wid.x;
  let tid = lid.x;

  for (var i = tid; i < N; i = i + WG) {
    sh[digitReverse(i)] = data[gidx(line, i)];
  }
  workgroupBarrier();

  var m: u32 = 1u;
  for (var s: u32 = 0u; s < R4_STAGES; s = s + 1u) {
    for (var bt = tid; bt < N / 4u; bt = bt + WG) {
      let j = bt % m;
      let base = (bt / m) * 4u * m + j;
      let ang = SGN * TWO_PI * f32(j) / f32(4u * m);
      let a0 = sh[base];
      let a1 = cmul(sh[base + m], twiddle(ang));
      let a2 = cmul(sh[base + 2u * m], twiddle(2.0 * ang));
      let a3 = cmul(sh[base + 3u * m], twiddle(3.0 * ang));
      let b0 = a0 + a2;
      let b1 = a0 - a2;
      let b2 = a1 + a3;
      let b3 = rot(a1 - a3);
      sh[base] = b0 + b2;
      sh[base + m] = b1 + b3;
      sh[base + 2u * m] = b0 - b2;
      sh[base + 3u * m] = b1 - b3;
    }
    workgroupBarrier();
    m = m * 4u;
  }

$R2_BLOCK$
  for (var i = tid; i < N; i = i + WG) {
    data[gidx(line, i)] = sh[i] * SCALE;
  }
}
