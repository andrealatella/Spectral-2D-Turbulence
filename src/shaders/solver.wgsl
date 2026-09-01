struct Rk {
  cA: f32,
  a: f32,
  cB: f32,
  b: f32,
}

@group(0) @binding(0) var<storage, read> w0: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> wcur: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> wout: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> packA: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> packB: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> physZ: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read_write> vel: array<vec2<f32>>;
@group(0) @binding(7) var<storage, read_write> rhs: array<vec2<f32>>;
@group(0) @binding(8) var<storage, read> fhat: array<vec2<f32>>;
@group(0) @binding(9) var<storage, read> dtbuf: array<f32>;
@group(0) @binding(10) var<uniform> P: Params;
@group(0) @binding(11) var<uniform> R: Rk;

@compute @workgroup_size(64)
fn velocity_pack(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= NCELLS) {
    return;
  }
  let kx = wn(idx % N);
  let ky = wn(idx / N);
  let k2 = kx * kx + ky * ky;
  var inv = 0.0;
  if (k2 > 0.0) {
    inv = 1.0 / k2;
  }
  let w = wcur[idx];
  let iw = cmuli(w);
  let uh = (ky * inv) * iw;
  let vh = (-kx * inv) * iw;
  packA[idx] = w + cmuli(uh);
  packB[idx] = vh;
}

@compute @workgroup_size(64)
fn nl_pack(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= NCELLS) {
    return;
  }
  let a = packA[idx];
  let omega = a.x;
  let u = a.y;
  let v = packB[idx].x;
  physZ[idx] = vec2<f32>(u * omega, v * omega);
  vel[idx] = vec2<f32>(u, v);
}

@compute @workgroup_size(64)
fn assemble(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= NCELLS) {
    return;
  }
  let col = idx % N;
  let row = idx / N;
  let kx = wn(col);
  let ky = wn(row);
  let k2 = kx * kx + ky * ky;

  let zk = physZ[idx];
  let zm = physZ[((N - row) % N) * N + ((N - col) % N)];
  let q1 = 0.5 * (zk + cconj(zm));
  let q2 = -0.5 * cmuli(zk - cconj(zm));
  let nhat = kx * cmuli(q1) + ky * cmuli(q2);

  let mask = inBand(kx, ky);
  var f = (fhat[idx] - nhat) * mask;

  if (P.forceAmp > 0.0 && mask > 0.0 && k2 > 0.0) {
    let env = exp(-pow((sqrt(k2) - P.forceK) / P.forceWidth, 2.0));
    if (env > 1e-4) {
      var hr = row;
      var hc = col;
      var sgn = 1.0;
      if (!(ky > 0.0 || (ky == 0.0 && kx > 0.0))) {
        hr = (N - row) % N;
        hc = (N - col) % N;
        sgn = -1.0;
      }
      let ph = TWO_PI * hashf(hr * N + hc, P.stepIndex);
      let amp = P.forceAmp * env * FORCE_NORM / sqrt(max(dtbuf[0], 1e-6));
      f = f + amp * vec2<f32>(cos(ph), sgn * sin(ph));
    }
  }

  rhs[idx] = f;
}

fn expL(kx: f32, k2: f32, inv: f32, s: f32) -> vec2<f32> {
  let re = -P.mu - P.nuH * pow(k2, P.hyperP);
  let im = P.beta * kx * inv;
  return exp(re * s) * vec2<f32>(cos(im * s), sin(im * s));
}

@compute @workgroup_size(64)
fn rk3(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= NCELLS) {
    return;
  }
  let kx = wn(idx % N);
  let ky = wn(idx / N);
  let k2 = kx * kx + ky * ky;
  if (k2 == 0.0) {
    wout[idx] = vec2<f32>(0.0, 0.0);
    return;
  }
  let inv = 1.0 / k2;
  let dt = dtbuf[0];
  let ea = expL(kx, k2, inv, R.a * dt);
  let eb = expL(kx, k2, inv, R.b * dt);
  let advanced = wcur[idx] + dt * rhs[idx];
  wout[idx] = R.cA * cmul(ea, w0[idx]) + R.cB * cmul(eb, advanced);
}
