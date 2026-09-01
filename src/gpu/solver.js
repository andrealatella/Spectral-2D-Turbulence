import { shaderModule, flt, gridConstants } from './shaders.js';
import { makeLayout, makeBindGroup, computePipeline, dispatch1d } from './pipeline.js';
import { Fft2D } from './fft.js';

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
const UNIFORM = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
const READBACK = GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST;

export const MAX_SPLATS = 8;
const RK_STRIDE = 256;
const RK_COEFFS = [
  [0, 0, 1, 1],
  [0.75, 0.5, 0.25, -0.5],
  [1 / 3, 1, 2 / 3, 0.5],
];

const F32 = new Float32Array(1);
const U32 = new Uint32Array(F32.buffer);

function toHalf(value) {
  F32[0] = value;
  const x = U32[0];
  const sign = (x >>> 16) & 0x8000;
  let exp = (x >>> 23) & 0xff;
  let mant = x & 0x7fffff;
  if (exp === 255) {
    return sign | 0x7c00 | (mant ? 0x200 : 0);
  }
  exp = exp - 127 + 15;
  if (exp >= 31) {
    return sign | 0x7c00;
  }
  if (exp <= 0) {
    if (exp < -10) {
      return sign;
    }
    mant = (mant | 0x800000) >> (1 - exp);
    return sign | (mant >> 13);
  }
  return sign | (exp << 10) | (mant >> 13);
}

function hash32(x) {
  let v = (Math.imul(x >>> 0, 747796405) + 2891336453) >>> 0;
  v = Math.imul((v >>> ((v >>> 28) + 4)) ^ v, 277803737) >>> 0;
  return ((v >>> 22) ^ v) >>> 0;
}

export class Solver {
  constructor(device, n, dyeScale) {
    this.device = device;
    this.n = n;
    this.ncells = n * n;
    this.m = n * dyeScale;
    this.nbins = n / 2 + 1;
    this.phase = 0;
    this.dyeParity = 0;
    this.stepIndex = 0;
    this.simTime = 0;
    this.readState = 'idle';
    this.disposed = false;

    this.params = new ArrayBuffer(48);
    this.paramsF = new Float32Array(this.params);
    this.paramsU = new Uint32Array(this.params);
    this.splatData = new ArrayBuffer(16 + MAX_SPLATS * 48);
    this.splatU = new Uint32Array(this.splatData, 0, 4);
    this.splatF = new Float32Array(this.splatData, 16);
  }

  static async create(device, { n = 256, dyeScale = 2 } = {}) {
    const s = new Solver(device, n, dyeScale);
    await s.build();
    return s;
  }

  async build() {
    const { device, n, m } = this;
    const complexBytes = n * n * 8;
    const consts = {
      ...gridConstants(n),
      NBINS: this.nbins,
      M: m,
      MF: flt(m),
    };

    const mk = (size, usage, label) => device.createBuffer({ size, usage, label });
    this.buf = {
      state: [0, 1, 2].map((i) => mk(complexBytes, STORAGE, `state${i}`)),
      packA: mk(complexBytes, STORAGE, 'packA'),
      packB: mk(complexBytes, STORAGE, 'packB'),
      physZ: mk(complexBytes, STORAGE, 'physZ'),
      rhs: mk(complexBytes, STORAGE, 'rhs'),
      force: mk(complexBytes, STORAGE, 'force'),
      vel: mk(complexBytes, STORAGE, 'vel'),
      dt: mk(16, STORAGE, 'dt'),
      partials: mk(256 * 4, STORAGE, 'partials'),
      bins: mk((this.nbins + 2) * 4, STORAGE, 'bins'),
      zonal: mk(n * 4, STORAGE, 'zonal'),
      params: mk(48, UNIFORM, 'params'),
      rk: mk(RK_STRIDE * 3, UNIFORM, 'rk'),
      splats: mk(this.splatData.byteLength, UNIFORM, 'splats'),
      dyeCfg: mk(16, UNIFORM, 'dyeCfg'),
    };
    this.staging = {
      bins: mk((this.nbins + 2) * 4, READBACK, 'bins-read'),
      zonal: mk(n * 4, READBACK, 'zonal-read'),
      dt: mk(16, READBACK, 'dt-read'),
    };

    const rk = new Float32Array((RK_STRIDE * 3) / 4);
    RK_COEFFS.forEach((c, i) => rk.set(c, (i * RK_STRIDE) / 4));
    device.queue.writeBuffer(this.buf.rk, 0, rk);

    this.dye = [0, 1, 2].map((i) =>
      device.createTexture({
        label: `dye${i}`,
        size: [m, m],
        format: 'rgba16float',
        usage:
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST,
      }),
    );
    this.sampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'repeat',
      addressModeV: 'repeat',
    });

    this.fft = await Fft2D.create(device, n);
    for (const name of ['packA', 'packB', 'physZ', 'force']) {
      this.fft.register(name, this.buf[name]);
    }

    await this.buildSolverStages(consts);
    await this.buildSupportStages(consts);
  }

  async buildSolverStages(consts) {
    const { device, buf } = this;
    const module = await shaderModule(device, 'solver', consts);

    const layouts = {
      velocityPack: makeLayout(device, [[1, 'read'], [3, 'rw'], [4, 'rw']], { label: 'velocity' }),
      nlPack: makeLayout(device, [[3, 'rw'], [4, 'rw'], [5, 'rw'], [6, 'rw']], { label: 'nl' }),
      assemble: makeLayout(
        device,
        [[5, 'rw'], [7, 'rw'], [8, 'read'], [9, 'read'], [10, 'uniform']],
        { label: 'assemble' },
      ),
      rk3: makeLayout(
        device,
        [
          [0, 'read'],
          [1, 'read'],
          [2, 'rw'],
          [7, 'rw'],
          [9, 'read'],
          [10, 'uniform'],
          [11, 'uniform-dyn'],
        ],
        { label: 'rk3' },
      ),
    };

    this.p = {
      velocityPack: computePipeline(device, module, 'velocity_pack', layouts.velocityPack),
      nlPack: computePipeline(device, module, 'nl_pack', layouts.nlPack),
      assemble: computePipeline(device, module, 'assemble', layouts.assemble),
      rk3: computePipeline(device, module, 'rk3', layouts.rk3),
    };

    this.bg = {
      velocityPack: this.buf.state.map((b, i) =>
        makeBindGroup(device, layouts.velocityPack, [[1, b], [3, buf.packA], [4, buf.packB]], `vp${i}`),
      ),
      nlPack: makeBindGroup(device, layouts.nlPack, [
        [3, buf.packA],
        [4, buf.packB],
        [5, buf.physZ],
        [6, buf.vel],
      ]),
      assemble: makeBindGroup(device, layouts.assemble, [
        [5, buf.physZ],
        [7, buf.rhs],
        [8, buf.force],
        [9, buf.dt],
        [10, buf.params],
      ]),
      rk3: [0, 1, 2].map((phase) =>
        [0, 1, 2].map((stage) => {
          const wcur = buf.state[(phase + stage) % 3];
          const wout = buf.state[(phase + (stage === 1 ? 2 : 1)) % 3];
          return makeBindGroup(
            device,
            layouts.rk3,
            [
              [0, buf.state[phase]],
              [1, wcur],
              [2, wout],
              [7, buf.rhs],
              [9, buf.dt],
              [10, buf.params],
              [11, { buffer: buf.rk, offset: 0, size: 16 }],
            ],
            `rk3-${phase}-${stage}`,
          );
        }),
      ),
    };
  }

  async buildSupportStages(consts) {
    const { device, buf } = this;

    const forceModule = await shaderModule(device, 'force', consts);
    const forceLayout = makeLayout(device, [[0, 'rw'], [1, 'uniform'], [2, 'uniform']]);
    this.p.force = computePipeline(device, forceModule, 'main', forceLayout);
    this.bg.force = makeBindGroup(device, forceLayout, [
      [0, buf.force],
      [1, buf.splats],
      [2, buf.params],
    ]);

    const reduceModule = await shaderModule(device, 'reduce', consts);
    const reduceLayout = makeLayout(device, [
      [0, 'read'],
      [1, 'rw'],
      [2, 'rw'],
      [3, 'uniform'],
    ]);
    this.p.reduceSpeed = computePipeline(device, reduceModule, 'reduce_speed', reduceLayout);
    this.p.reduceDt = computePipeline(device, reduceModule, 'reduce_dt', reduceLayout);
    this.bg.reduce = makeBindGroup(device, reduceLayout, [
      [0, buf.vel],
      [1, buf.partials],
      [2, buf.dt],
      [3, buf.params],
    ]);

    const specModule = await shaderModule(device, 'spectrum', consts);
    const specLayout = makeLayout(device, [
      [0, 'read'],
      [1, 'rw'],
      [2, 'read'],
      [3, 'rw'],
    ]);
    this.p.clearBins = computePipeline(device, specModule, 'clear_bins', specLayout);
    this.p.accumulate = computePipeline(device, specModule, 'accumulate', specLayout);
    this.p.zonalMean = computePipeline(device, specModule, 'zonal_mean', specLayout);
    this.bg.spectrum = buf.state.map((b, i) =>
      makeBindGroup(
        device,
        specLayout,
        [[0, b], [1, buf.bins], [2, buf.packA], [3, buf.zonal]],
        `spec${i}`,
      ),
    );

    const seedModule = await shaderModule(device, 'seed', consts);
    const seedLayout = makeLayout(device, [[0, 'read'], [1, 'rw']]);
    this.p.seedMask = computePipeline(device, seedModule, 'main', seedLayout);
    this.bg.seedMask = makeBindGroup(device, seedLayout, [[0, buf.packA], [1, buf.state[0]]]);

    const dyeModule = await shaderModule(device, 'dye', consts);
    const dyeLayout = makeLayout(device, [
      [0, 'read'],
      [1, 'tex'],
      [2, 'tex'],
      [3, 'storage-tex'],
      [4, 'sampler'],
      [5, 'read'],
      [6, 'uniform'],
      [7, 'uniform'],
    ]);
    this.p.dyeFwd = computePipeline(device, dyeModule, 'advect_fwd', dyeLayout);
    this.p.dyeBack = computePipeline(device, dyeModule, 'advect_back', dyeLayout);
    this.p.dyeCorr = computePipeline(device, dyeModule, 'advect_corr', dyeLayout);

    const dyeGroup = (src, aux, dst, label) =>
      makeBindGroup(
        device,
        dyeLayout,
        [
          [0, buf.vel],
          [1, this.dye[src]],
          [2, this.dye[aux]],
          [3, this.dye[dst]],
          [4, this.sampler],
          [5, buf.dt],
          [6, buf.splats],
          [7, buf.dyeCfg],
        ],
        label,
      );
    this.bg.dye = [0, 1].map((p) => ({
      fwd: dyeGroup(p, p, 1 - p, `dye-fwd-${p}`),
      back: dyeGroup(1 - p, p, 2, `dye-back-${p}`),
      corr: dyeGroup(p, 2, 1 - p, `dye-corr-${p}`),
    }));
  }

  configure(cfg) {
    const kcut = this.n / 3;
    const f = this.paramsF;
    f[0] = cfg.beta;
    f[1] = cfg.drag;
    f[2] = cfg.viscosity > 0 ? cfg.viscosity / kcut ** (2 * cfg.hyperP) : 0;
    f[3] = cfg.hyperP;
    f[4] = cfg.forceAmp;
    f[5] = cfg.forceK;
    f[6] = cfg.forceWidth;
    this.paramsU[7] = this.stepIndex >>> 0;
    f[8] = (2 * Math.PI) / this.n;
    f[9] = cfg.cfl;
    f[10] = cfg.dtMin;
    f[11] = cfg.dtMax;
    this.device.queue.writeBuffer(this.buf.params, 0, this.params);
    this.device.queue.writeBuffer(this.buf.dyeCfg, 0, new Float32Array([cfg.dyeFade, 0, 0, 0]));
    this.cfg = cfg;
  }

  setSplats(list) {
    const count = Math.min(list.length, MAX_SPLATS);
    this.splatU[0] = count;
    this.splatF.fill(0);
    for (let i = 0; i < count; i++) {
      const s = list[i];
      const o = i * 12;
      this.splatF[o] = s.x;
      this.splatF[o + 1] = s.y;
      this.splatF[o + 2] = s.dx;
      this.splatF[o + 3] = s.dy;
      this.splatF[o + 4] = s.color[0];
      this.splatF[o + 5] = s.color[1];
      this.splatF[o + 6] = s.color[2];
      this.splatF[o + 7] = s.radius;
      this.splatF[o + 8] = s.amp;
      this.splatF[o + 9] = s.dyeAmp;
    }
    this.device.queue.writeBuffer(this.buf.splats, 0, this.splatData);
  }

  seed({ k0 = 14, amplitude = 6, seed = 1 } = {}) {
    const { n } = this;
    const data = new Float32Array(2 * n * n);
    if (amplitude > 0) {
      this.fillSpectrum(data, k0, amplitude, seed);
    }

    this.phase = 0;
    this.device.queue.writeBuffer(this.buf.state[0], 0, data);
    for (const other of [1, 2]) {
      this.device.queue.writeBuffer(this.buf.state[other], 0, new Float32Array(2 * n * n));
    }
  }

  seedPhysical(values) {
    const { n, device } = this;
    const data = new Float32Array(2 * n * n);
    for (let i = 0; i < n * n; i++) {
      data[2 * i] = values[i];
    }
    device.queue.writeBuffer(this.buf.packA, 0, data);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    this.fft.forward(pass, 'packA');
    pass.setPipeline(this.p.seedMask);
    pass.setBindGroup(0, this.bg.seedMask);
    dispatch1d(pass, this.ncells);
    pass.end();
    device.queue.submit([encoder.finish()]);

    this.phase = 0;
    for (const other of [1, 2]) {
      device.queue.writeBuffer(this.buf.state[other], 0, new Float32Array(2 * n * n));
    }
  }

  fillSpectrum(data, k0, amplitude, seed) {
    const { n } = this;
    const kcut = n / 3;
    const salt = hash32(seed);
    let sum = 0;

    for (let idx = 0; idx < n * n; idx++) {
      const col = idx % n;
      const row = (idx / n) | 0;
      const kx = col < n / 2 ? col : col - n;
      const ky = row < n / 2 ? row : row - n;
      if ((kx === 0 && ky === 0) || Math.abs(kx) >= kcut || Math.abs(ky) >= kcut) {
        continue;
      }
      const kr = Math.hypot(kx, ky);
      const env = (kr * kr) / (1 + (kr / k0) ** 8);
      const amp = Math.sqrt(env);
      const canonical = ky > 0 || (ky === 0 && kx > 0);
      const hr = canonical ? row : (n - row) % n;
      const hc = canonical ? col : (n - col) % n;
      const sgn = canonical ? 1 : -1;
      const phase = (2 * Math.PI * hash32((hr * n + hc) ^ salt)) / 4294967296;
      data[2 * idx] = amp * Math.cos(phase);
      data[2 * idx + 1] = sgn * amp * Math.sin(phase);
      sum += amp * amp;
    }

    const rms = Math.sqrt(sum) / (n * n);
    if (rms > 0) {
      const scale = amplitude / rms;
      for (let i = 0; i < data.length; i++) {
        data[i] *= scale;
      }
    }
  }

  reset(opts = {}) {
    this.stepIndex = 0;
    this.simTime = 0;
    this.dyeParity = 0;
    if (opts.field) {
      this.seedPhysical(opts.field);
    } else {
      this.seed(opts);
    }
    const blank = new Uint8Array(this.m * this.m * 8);
    for (const tex of this.dye) {
      this.device.queue.writeTexture({ texture: tex }, blank, { bytesPerRow: this.m * 8 }, [
        this.m,
        this.m,
      ]);
    }
    this.paintDye(opts?.dyePattern ?? 'bands');
    this.device.queue.writeBuffer(
      this.buf.dt,
      0,
      new Float32Array([this.cfg?.dtMax ?? 0.01, 0, 0, 0]),
    );
    this.device.queue.writeBuffer(this.buf.force, 0, new Float32Array(2 * this.n * this.n));
    this.primed = false;
  }

  paintDye(pattern) {
    if (pattern === 'none') {
      return;
    }
    const m = this.m;
    const data = new Uint16Array(m * m * 4);
    const warm = [1.0, 0.45, 0.22];
    const cool = [0.28, 0.62, 1.0];
    for (let row = 0; row < m; row++) {
      const fy = row / m;
      const band = 0.5 + 0.5 * Math.tanh(6 * Math.sin(fy * Math.PI * 12));
      for (let col = 0; col < m; col++) {
        const fx = col / m;
        const mix = 0.5 + 0.5 * Math.sin(fx * Math.PI * 2);
        const o = (row * m + col) * 4;
        for (let c = 0; c < 3; c++) {
          data[o + c] = toHalf(band * (warm[c] * mix + cool[c] * (1 - mix)) * 0.9);
        }
        data[o + 3] = toHalf(1);
      }
    }
    this.device.queue.writeTexture(
      { texture: this.dye[this.dyeParity] },
      data,
      { bytesPerRow: m * 8 },
      [m, m],
    );
  }

  encodePrime(pass) {
    pass.setPipeline(this.p.velocityPack);
    pass.setBindGroup(0, this.bg.velocityPack[this.phase]);
    dispatch1d(pass, this.ncells);
    this.fft.inverse(pass, 'packA');
    this.fft.inverse(pass, 'packB');
    pass.setPipeline(this.p.nlPack);
    pass.setBindGroup(0, this.bg.nlPack);
    dispatch1d(pass, this.ncells);
    this.encodeReduce(pass);
    this.primed = true;
  }

  encodeStep(pass) {
    const phase = this.phase;

    pass.setPipeline(this.p.force);
    pass.setBindGroup(0, this.bg.force);
    dispatch1d(pass, this.ncells);
    this.fft.forward(pass, 'force');

    for (let stage = 0; stage < 3; stage++) {
      pass.setPipeline(this.p.velocityPack);
      pass.setBindGroup(0, this.bg.velocityPack[(phase + stage) % 3]);
      dispatch1d(pass, this.ncells);

      this.fft.inverse(pass, 'packA');
      this.fft.inverse(pass, 'packB');

      pass.setPipeline(this.p.nlPack);
      pass.setBindGroup(0, this.bg.nlPack);
      dispatch1d(pass, this.ncells);

      this.fft.forward(pass, 'physZ');

      pass.setPipeline(this.p.assemble);
      pass.setBindGroup(0, this.bg.assemble);
      dispatch1d(pass, this.ncells);

      pass.setPipeline(this.p.rk3);
      pass.setBindGroup(0, this.bg.rk3[phase][stage], [stage * RK_STRIDE]);
      dispatch1d(pass, this.ncells);
    }

    this.phase = (phase + 1) % 3;
    this.stepIndex += 1;
  }

  encodeDye(pass) {
    const g = this.bg.dye[this.dyeParity];
    const groups = Math.ceil(this.m / 8);
    for (const [pipeline, group] of [
      [this.p.dyeFwd, g.fwd],
      [this.p.dyeBack, g.back],
      [this.p.dyeCorr, g.corr],
    ]) {
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(groups, groups);
    }
    this.dyeParity = 1 - this.dyeParity;
  }

  encodeReduce(pass) {
    pass.setPipeline(this.p.reduceSpeed);
    pass.setBindGroup(0, this.bg.reduce);
    pass.dispatchWorkgroups(256);
    pass.setPipeline(this.p.reduceDt);
    pass.setBindGroup(0, this.bg.reduce);
    pass.dispatchWorkgroups(1);
  }

  encodeDiagnostics(pass) {
    const group = this.bg.spectrum[this.phase];
    pass.setPipeline(this.p.clearBins);
    pass.setBindGroup(0, group);
    dispatch1d(pass, this.nbins + 2);
    pass.setPipeline(this.p.accumulate);
    pass.setBindGroup(0, group);
    dispatch1d(pass, this.ncells);
    pass.setPipeline(this.p.zonalMean);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(this.n);
  }

  copyDiagnostics(encoder) {
    if (this.readState !== 'idle') {
      return;
    }
    encoder.copyBufferToBuffer(this.buf.bins, 0, this.staging.bins, 0, (this.nbins + 2) * 4);
    encoder.copyBufferToBuffer(this.buf.zonal, 0, this.staging.zonal, 0, this.n * 4);
    encoder.copyBufferToBuffer(this.buf.dt, 0, this.staging.dt, 0, 16);
    this.readState = 'copied';
  }

  async readDiagnostics() {
    if (this.readState !== 'copied' || this.disposed) {
      return null;
    }
    this.readState = 'mapping';
    try {
      await Promise.all([
        this.staging.bins.mapAsync(GPUMapMode.READ),
        this.staging.zonal.mapAsync(GPUMapMode.READ),
        this.staging.dt.mapAsync(GPUMapMode.READ),
      ]);
      if (this.disposed) {
        return null;
      }
      const raw = new Uint32Array(this.staging.bins.getMappedRange()).slice();
      const zonal = new Float32Array(this.staging.zonal.getMappedRange()).slice();
      const dt = new Float32Array(this.staging.dt.getMappedRange()).slice();
      this.staging.bins.unmap();
      this.staging.zonal.unmap();
      this.staging.dt.unmap();

      const spectrum = new Float32Array(this.nbins);
      for (let i = 0; i < this.nbins; i++) {
        spectrum[i] = raw[i] / 1e8;
      }
      return {
        spectrum,
        zonal,
        energy: raw[this.nbins] / 1e6,
        enstrophy: raw[this.nbins + 1] / 1e6,
        dt: dt[0],
        maxSpeed: dt[1],
      };
    } catch {
      return null;
    } finally {
      this.readState = 'idle';
    }
  }

  get currentDye() {
    return this.dye[this.dyeParity];
  }

  destroy() {
    this.disposed = true;
    for (const b of Object.values(this.buf)) {
      if (Array.isArray(b)) {
        b.forEach((x) => x.destroy());
      } else {
        b.destroy();
      }
    }
    for (const b of Object.values(this.staging)) {
      b.destroy();
    }
    for (const t of this.dye) {
      t.destroy();
    }
  }
}
