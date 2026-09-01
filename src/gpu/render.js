import { shaderModule, gridConstants } from './shaders.js';
import { makeLayout, makeBindGroup } from './pipeline.js';

export const MODES = ['dye', 'vorticity', 'dye over vorticity', 'speed'];

export class Renderer {
  constructor(device, context, pipeline, groups, cfgBuffer) {
    this.device = device;
    this.context = context;
    this.pipeline = pipeline;
    this.groups = groups;
    this.cfgBuffer = cfgBuffer;
    this.cfg = new ArrayBuffer(32);
    this.cfgF = new Float32Array(this.cfg);
    this.cfgU = new Uint32Array(this.cfg);
  }

  static async create(device, canvas, solver, format) {
    const context = canvas.getContext('webgpu');
    context.configure({ device, format, alphaMode: 'opaque' });

    const module = await shaderModule(device, 'render', gridConstants(solver.n));

    const layout = makeLayout(
      device,
      [
        [0, 'tex'],
        [1, 'sampler'],
        [2, 'read'],
        [3, 'read'],
        [4, 'uniform'],
      ],
      { visibility: GPUShaderStage.FRAGMENT, label: 'render' },
    );

    const pipeline = device.createRenderPipeline({
      label: 'present',
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    });

    const cfgBuffer = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const groups = [0, 1].map((parity) =>
      makeBindGroup(
        device,
        layout,
        [
          [0, solver.dye[parity]],
          [1, solver.sampler],
          [2, solver.buf.packA],
          [3, solver.buf.vel],
          [4, cfgBuffer],
        ],
        `present-${parity}`,
      ),
    );

    return new Renderer(device, context, pipeline, groups, cfgBuffer);
  }

  render(encoder, solver, cfg) {
    this.cfgU[0] = MODES.indexOf(cfg.mode) < 0 ? 0 : MODES.indexOf(cfg.mode);
    this.cfgF[2] = cfg.exposure;
    this.cfgF[3] = cfg.vortScale;
    this.cfgF[4] = cfg.dyeGain;
    this.device.queue.writeBuffer(this.cfgBuffer, 0, this.cfg);

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          clearValue: { r: 0.02, g: 0.02, b: 0.03, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.groups[solver.dyeParity]);
    pass.draw(3);
    pass.end();
  }
}
