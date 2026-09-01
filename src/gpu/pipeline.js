const STORAGE_FORMAT = 'rgba16float';

function layoutEntry(binding, type, visibility) {
  switch (type) {
    case 'uniform':
      return { binding, visibility, buffer: { type: 'uniform' } };
    case 'uniform-dyn':
      return { binding, visibility, buffer: { type: 'uniform', hasDynamicOffset: true } };
    case 'read':
      return { binding, visibility, buffer: { type: 'read-only-storage' } };
    case 'rw':
      return { binding, visibility, buffer: { type: 'storage' } };
    case 'tex':
      return { binding, visibility, texture: { sampleType: 'float' } };
    case 'storage-tex':
      return {
        binding,
        visibility,
        storageTexture: { access: 'write-only', format: STORAGE_FORMAT },
      };
    case 'sampler':
      return { binding, visibility, sampler: { type: 'filtering' } };
    default:
      throw new Error(`unknown binding type ${type}`);
  }
}

export function makeLayout(device, spec, { visibility = GPUShaderStage.COMPUTE, label } = {}) {
  return device.createBindGroupLayout({
    label,
    entries: spec.map(([binding, type]) => layoutEntry(binding, type, visibility)),
  });
}

export function makeBindGroup(device, layout, spec, label) {
  return device.createBindGroup({
    label,
    layout,
    entries: spec.map(([binding, resource]) => ({
      binding,
      resource:
        resource instanceof GPUBuffer
          ? { buffer: resource }
          : resource instanceof GPUTexture
            ? resource.createView()
            : resource,
    })),
  });
}

export function computePipeline(device, module, entryPoint, layout, label) {
  return device.createComputePipeline({
    label: label ?? entryPoint,
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    compute: { module, entryPoint },
  });
}

export function dispatch1d(pass, count, workgroupSize = 64) {
  pass.dispatchWorkgroups(Math.ceil(count / workgroupSize));
}
