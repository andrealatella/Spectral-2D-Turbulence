const N: u32 = $N$u;
const NF: f32 = $NF$;
const NCELLS: u32 = $NCELLS$u;
const KCUT: f32 = $KCUT$;
const TWO_PI: f32 = 6.28318530717958647;
const FORCE_NORM: f32 = $FORCE_NORM$;

struct Params {
  beta: f32,
  mu: f32,
  nuH: f32,
  hyperP: f32,
  forceAmp: f32,
  forceK: f32,
  forceWidth: f32,
  stepIndex: u32,
  dx: f32,
  cfl: f32,
  dtMin: f32,
  dtMax: f32,
}

struct Splat {
  pos: vec2<f32>,
  vel: vec2<f32>,
  color: vec3<f32>,
  radius: f32,
  amp: f32,
  dyeAmp: f32,
  pad0: f32,
  pad1: f32,
}

struct Splats {
  count: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
  items: array<Splat, 8>,
}

fn wrap01(x: f32) -> f32 {
  return x - floor(x);
}

fn sep(a: f32, b: f32) -> f32 {
  return wrap01(a - b + 0.5) - 0.5;
}

fn wn(i: u32) -> f32 {
  if (i < N / 2u) {
    return f32(i);
  }
  return f32(i) - NF;
}

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn cmuli(z: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(-z.y, z.x);
}

fn cconj(z: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(z.x, -z.y);
}

fn inBand(kx: f32, ky: f32) -> f32 {
  if (abs(kx) < KCUT && abs(ky) < KCUT) {
    return 1.0;
  }
  return 0.0;
}

fn hash1(x: u32) -> u32 {
  var v = x * 747796405u + 2891336453u;
  v = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
  return (v >> 22u) ^ v;
}

fn hashf(a: u32, b: u32) -> f32 {
  return f32(hash1(a ^ hash1(b))) * 2.3283064365386963e-10;
}
