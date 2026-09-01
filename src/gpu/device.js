const WANTED_LIMITS = [
  'maxComputeWorkgroupStorageSize',
  'maxComputeInvocationsPerWorkgroup',
  'maxStorageBufferBindingSize',
  'maxBufferSize',
];

export async function initDevice() {
  if (!navigator.gpu) {
    throw new Error(
      'WebGPU is unavailable. Use Chrome or Edge 113+, Firefox 141+ on Windows, or Safari 26.',
    );
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    throw new Error('No WebGPU adapter was returned. GPU access may be blocked or blocklisted.');
  }

  const requiredLimits = {};
  for (const key of WANTED_LIMITS) {
    const value = adapter.limits[key];
    if (typeof value === 'number') {
      requiredLimits[key] = value;
    }
  }

  const device = await adapter.requestDevice({ label: 'spectral-solver', requiredLimits });
  device.lost.then((info) => {
    if (info.reason !== 'destroyed') {
      console.error('WebGPU device lost:', info.message);
    }
  });

  const info = adapter.info ?? {};
  return {
    device,
    adapter,
    description: [info.vendor, info.architecture, info.device].filter(Boolean).join(' ') || 'unknown GPU',
    grids: supportedGrids(device.limits),
  };
}

export function supportedGrids(limits) {
  const maxBuffer = Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize);
  return [128, 256, 512, 1024].filter(
    (n) =>
      n * 8 <= limits.maxComputeWorkgroupStorageSize &&
      Math.min(n / 4, 256) <= limits.maxComputeInvocationsPerWorkgroup &&
      n * n * 8 <= maxBuffer,
  );
}
