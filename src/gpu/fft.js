import { shaderModule, flt } from './shaders.js';
import { makeLayout, makeBindGroup, computePipeline } from './pipeline.js';

const R2_BLOCK = `  for (var bt = tid; bt < N / 2u; bt = bt + WG) {
    let j = bt % m;
    let base = (bt / m) * 2u * m + j;
    let ang = SGN * TWO_PI * f32(j) / f32(2u * m);
    let a0 = sh[base];
    let a1 = cmul(sh[base + m], twiddle(ang));
    sh[base] = a0 + a1;
    sh[base + m] = a0 - a1;
  }
  workgroupBarrier();
`;

const REV_R2 = `  out = out * 2u + (v % 2u);
  v = v / 2u;
`;

export function radixSchedule(n) {
  let stages = 0;
  let m = n;
  while (m % 4 === 0) {
    stages += 1;
    m /= 4;
  }
  if (m !== 1 && m !== 2) {
    throw new Error(`grid size ${n} must be 4^a or 2 * 4^a`);
  }
  return { stages, hasRadix2: m === 2 };
}

export class Fft2D {
  constructor(device, n, layout, pipelines) {
    this.device = device;
    this.n = n;
    this.layout = layout;
    this.pipelines = pipelines;
    this.groups = new Map();
  }

  static async create(device, n) {
    const wg = Math.min(n / 4, 256);
    const { stages, hasRadix2 } = radixSchedule(n);
    const layout = makeLayout(device, [[0, 'rw']], { label: 'fft-layout' });
    const pipelines = {};

    for (const axis of [0, 1]) {
      for (const inverse of [0, 1]) {
        const module = await shaderModule(
          device,
          'fft',
          {
            N: n,
            WG: wg,
            AXIS: axis,
            R4_STAGES: stages,
            SGN: flt(inverse ? 1 : -1),
            SCALE: flt(inverse ? 1 / n : 1),
            R2_BLOCK: hasRadix2 ? R2_BLOCK : '',
            REV_R2: hasRadix2 ? REV_R2 : '',
          },
          { prelude: false },
        );
        pipelines[`${axis}${inverse}`] = computePipeline(
          device,
          module,
          'main',
          layout,
          `fft-axis${axis}-${inverse ? 'inv' : 'fwd'}`,
        );
      }
    }
    return new Fft2D(device, n, layout, pipelines);
  }

  register(name, buffer) {
    this.groups.set(name, makeBindGroup(this.device, this.layout, [[0, buffer]], `fft-${name}`));
  }

  run(pass, name, inverse) {
    const group = this.groups.get(name);
    if (!group) {
      throw new Error(`buffer ${name} was never registered with the FFT`);
    }
    const dir = inverse ? 1 : 0;
    for (const axis of [0, 1]) {
      pass.setPipeline(this.pipelines[`${axis}${dir}`]);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(this.n);
    }
  }

  forward(pass, name) {
    this.run(pass, name, false);
  }

  inverse(pass, name) {
    this.run(pass, name, true);
  }
}
