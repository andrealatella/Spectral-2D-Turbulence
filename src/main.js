import { initDevice } from './gpu/device.js';
import { Solver } from './gpu/solver.js';
import { Renderer, MODES } from './gpu/render.js';
import { Pointer } from './ui/pointer.js';
import { buildControls } from './ui/controls.js';
import { drawSpectrum, drawZonal } from './ui/plots.js';
import { SCENARIOS, SCENARIO_NAMES } from './scenarios.js';

const config = {
  resolution: 256,
  dyeScale: 2,
  beta: 0,
  drag: 0.01,
  viscosity: 0.67,
  hyperP: 4,
  forceAmp: 0,
  forceK: 22,
  forceWidth: 2.5,
  cfl: 0.5,
  dtMin: 1e-5,
  dtMax: 0.02,
  substeps: 1,
  dyeFade: 0.999,
  stirStrength: 400,
  stirRadius: 0.045,
  seedScale: 14,
  seedAmplitude: 6,
  scenario: 'random',
  polygonN: 6,
  mergerSep: 0.7,
  mode: 'dye over vorticity',
  exposure: 1.3,
  vortScale: 0.06,
  dyeGain: 1.0,
  paused: false,
};

const PRESETS = {
  'free turbulence': {
    beta: 0,
    drag: 0.004,
    viscosity: 0.67,
    hyperP: 4,
    forceAmp: 0,
    seedScale: 22,
    seedAmplitude: 8,
    mode: 'dye over vorticity',
    vortScale: 0.05,
  },
  'inverse cascade': {
    beta: 0,
    drag: 0.03,
    viscosity: 1.0,
    hyperP: 4,
    forceAmp: 6,
    forceK: 26,
    forceWidth: 2.5,
    seedScale: 26,
    seedAmplitude: 2,
    mode: 'vorticity',
    vortScale: 0.05,
  },
  'jupiter bands': {
    beta: 34,
    drag: 0.02,
    viscosity: 1.0,
    hyperP: 4,
    forceAmp: 7,
    forceK: 24,
    forceWidth: 2.5,
    seedScale: 24,
    seedAmplitude: 3,
    mode: 'dye over vorticity',
    vortScale: 0.05,
  },
  'vortex merging': {
    beta: 0,
    drag: 0.0,
    viscosity: 0.25,
    hyperP: 4,
    forceAmp: 0,
    seedScale: 8,
    seedAmplitude: 7,
    mode: 'vorticity',
    vortScale: 0.08,
  },
  'still euler': {
    beta: 0,
    drag: 0,
    viscosity: 0,
    hyperP: 1,
    forceAmp: 0,
    seedAmplitude: 0,
    mode: 'dye over vorticity',
    vortScale: 0.08,
    stirStrength: 700,
  },
  'still viscous': {
    beta: 0,
    drag: 0,
    viscosity: 2.5,
    hyperP: 1,
    forceAmp: 0,
    seedAmplitude: 0,
    mode: 'dye over vorticity',
    vortScale: 0.08,
    stirStrength: 700,
  },
};

const VISC_MIN = 0.02;
const VISC_MAX = 400;
const VISC_STEPS = 240;

function viscFromSlider(i) {
  if (i <= 0) {
    return 0;
  }
  return VISC_MIN * (VISC_MAX / VISC_MIN) ** ((i - 1) / (VISC_STEPS - 1));
}

function viscToSlider(v) {
  if (!(v > 0)) {
    return 0;
  }
  const t = Math.log(v / VISC_MIN) / Math.log(VISC_MAX / VISC_MIN);
  return Math.max(1, Math.min(VISC_STEPS, Math.round(1 + t * (VISC_STEPS - 1))));
}

function viscosityCoefficient(rate) {
  return rate / (config.resolution / 3) ** (2 * config.hyperP);
}

const SCHEMA = [
  {
    title: 'physics',
    items: [
      {
        type: 'range',
        key: 'viscosity',
        label: 'viscosity',
        min: 0,
        max: VISC_STEPS,
        step: 1,
        map: viscFromSlider,
        unmap: viscToSlider,
        format: (v) => (v > 0 ? `nu ${viscosityCoefficient(v).toExponential(1)}` : 'inviscid (Euler)'),
      },
      {
        type: 'select',
        key: 'hyperP',
        label: 'dissipation order',
        numeric: true,
        options: [
          { value: 1, label: '1  Navier-Stokes' },
          { value: 2, label: '2  biharmonic' },
          { value: 4, label: '4  hyperviscous' },
        ],
      },
      { type: 'range', key: 'beta', label: 'beta (Coriolis)', min: 0, max: 80, step: 1 },
      { type: 'range', key: 'drag', label: 'large-scale drag', min: 0, max: 0.1, step: 0.001 },
    ],
  },
  {
    title: 'scenario',
    items: [
      {
        type: 'range',
        key: 'polygonN',
        label: 'polygon vortices',
        min: 3,
        max: 9,
        step: 1,
        format: (v) => `${v}  ${v <= 6 ? 'stable' : v === 7 ? 'marginal' : 'unstable'}`,
      },
      { type: 'range', key: 'mergerSep', label: 'merger separation', min: 0.3, max: 2.4, step: 0.05 },
    ],
  },
  {
    title: 'forcing',
    items: [
      { type: 'range', key: 'forceAmp', label: 'stochastic amplitude', min: 0, max: 20, step: 0.5 },
      { type: 'range', key: 'forceK', label: 'forcing wavenumber', min: 4, max: 60, step: 1 },
      { type: 'range', key: 'forceWidth', label: 'forcing bandwidth', min: 0.5, max: 8, step: 0.5 },
      { type: 'range', key: 'stirStrength', label: 'pointer strength', min: 0, max: 2000, step: 25 },
      { type: 'range', key: 'stirRadius', label: 'pointer radius', min: 0.01, max: 0.15, step: 0.005 },
    ],
  },
  {
    title: 'integration',
    items: [
      { type: 'range', key: 'cfl', label: 'CFL number', min: 0.1, max: 1.2, step: 0.05 },
      { type: 'range', key: 'dtMax', label: 'max timestep', min: 0.002, max: 0.05, step: 0.001 },
      { type: 'select', key: 'substeps', label: 'steps per frame', options: [1, 2, 4, 8], numeric: true },
      { type: 'select', key: 'resolution', label: 'grid', options: [128, 256, 512], numeric: true, dynamic: 'grids' },
      { type: 'select', key: 'dyeScale', label: 'dye refinement', options: [1, 2], numeric: true },
    ],
  },
  {
    title: 'display',
    items: [
      { type: 'select', key: 'mode', label: 'field', options: MODES },
      { type: 'range', key: 'exposure', label: 'exposure', min: 0.2, max: 3, step: 0.05 },
      { type: 'range', key: 'vortScale', label: 'vorticity scale', min: 0.005, max: 0.3, step: 0.005 },
      { type: 'range', key: 'dyeGain', label: 'dye gain', min: 0, max: 3, step: 0.05 },
      { type: 'range', key: 'dyeFade', label: 'dye persistence', min: 0.99, max: 1, step: 0.0005, format: (v) => v.toFixed(4) },
    ],
  },
  {
    title: 'actions',
    items: [
      { type: 'button', key: 'still', label: 'still the fluid' },
      { type: 'button', key: 'reset', label: 'reseed turbulence' },
      { type: 'button', key: 'clearDye', label: 'repaint dye' },
      { type: 'button', key: 'pause', label: 'pause / resume' },
    ],
  },
];

const dom = {
  canvas: document.getElementById('sim'),
  panel: document.getElementById('panel'),
  stats: document.getElementById('stats'),
  spectrum: document.getElementById('spectrum'),
  zonal: document.getElementById('zonal'),
  presets: document.getElementById('presets'),
  scenarios: document.getElementById('scenarios'),
  status: document.getElementById('status'),
};

function fail(message) {
  dom.status.textContent = message;
  dom.status.classList.add('visible', 'error');
  console.error(message);
}

async function boot() {
  let gpu;
  try {
    gpu = await initDevice();
  } catch (err) {
    fail(err.message);
    return;
  }

  const { device } = gpu;
  const format = navigator.gpu.getPreferredCanvasFormat();

  const gridItem = SCHEMA.flatMap((g) => g.items).find((i) => i.dynamic === 'grids');
  gridItem.options = gpu.grids;
  if (!gpu.grids.includes(config.resolution)) {
    config.resolution = gpu.grids.includes(256) ? 256 : gpu.grids[gpu.grids.length - 1];
  }

  const pointer = new Pointer(dom.canvas);
  let solver = null;
  let renderer = null;
  let diagnostics = null;
  let hover = null;
  let rebuilding = false;

  async function rebuild() {
    rebuilding = true;
    if (solver) {
      solver.destroy();
    }
    solver = await Solver.create(device, {
      n: config.resolution,
      dyeScale: config.dyeScale,
    });
    solver.configure(config);
    reseed(config.seedAmplitude);
    renderer = await Renderer.create(device, dom.canvas, solver, format);
    diagnostics = null;
    rebuilding = false;
  }

  let pendingReseed = false;

  const reseed = (amplitude) => {
    const scenario = SCENARIOS[config.scenario];
    if (scenario) {
      solver.reset({ field: scenario.build(config.resolution, config) });
    } else {
      solver.reset({ k0: config.seedScale, amplitude, seed: Date.now() & 0xffff });
    }
  };

  const controls = buildControls(dom.panel, SCHEMA, config, async (key) => {
    if (key === 'resolution' || key === 'dyeScale') {
      await rebuild();
      controls.sync();
      return;
    }
    if (key === 'hyperP') {
      controls.sync();
      return;
    }
    if (key === 'polygonN' || key === 'mergerSep') {
      if (config.scenario === 'polygon' || config.scenario === 'merger') {
        pendingReseed = true;
      }
      return;
    }
    if (key === 'still') {
      config.scenario = 'random';
      reseed(0);
      return;
    }
    if (key === 'reset') {
      config.scenario = 'random';
      reseed(config.seedAmplitude || 6);
      return;
    }
    if (key === 'clearDye') {
      solver.paintDye('bands');
      return;
    }
    if (key === 'pause') {
      config.paused = !config.paused;
    }
  });

  for (const name of Object.keys(PRESETS)) {
    const button = document.createElement('button');
    button.className = 'preset';
    button.textContent = name;
    button.addEventListener('click', () => {
      Object.assign(config, PRESETS[name], { scenario: 'random' });
      controls.sync();
      solver.configure(config);
      reseed(config.seedAmplitude);
    });
    dom.presets.appendChild(button);
  }

  for (const name of SCENARIO_NAMES) {
    const button = document.createElement('button');
    button.className = 'preset';
    button.textContent = name;
    button.addEventListener('click', () => {
      Object.assign(config, SCENARIOS[name].config, { scenario: name });
      controls.sync();
      solver.configure(config);
      solver.paintDye('bands');
      reseed(config.seedAmplitude);
    });
    dom.scenarios.appendChild(button);
  }

  dom.spectrum.addEventListener('pointermove', (e) => {
    const r = dom.spectrum.getBoundingClientRect();
    hover = { x: e.clientX - r.left };
  });
  dom.spectrum.addEventListener('pointerleave', () => {
    hover = null;
  });

  window.addEventListener('keydown', (e) => {
    if (e.key === ' ') {
      e.preventDefault();
      config.paused = !config.paused;
    } else if (e.key === 'r') {
      reseed(config.seedAmplitude || 6);
    } else if (e.key === 's') {
      reseed(0);
    }
  });

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = dom.canvas.getBoundingClientRect();
    const size = Math.max(1, Math.round(Math.min(rect.width, rect.height) * dpr));
    if (dom.canvas.width !== size) {
      dom.canvas.width = size;
      dom.canvas.height = size;
    }
  }
  window.addEventListener('resize', resize);

  await rebuild();
  resize();

  let frames = 0;
  let fps = 0;
  let lastFpsAt = performance.now();
  let tick = 0;

  function loop() {
    requestAnimationFrame(loop);
    if (rebuilding || !solver || !renderer) {
      return;
    }
    resize();

    if (pendingReseed) {
      pendingReseed = false;
      reseed(config.seedAmplitude);
    }

    pointer.strength = config.stirStrength;
    pointer.radius = config.stirRadius;
    solver.configure(config);
    solver.setSplats(config.paused ? [] : pointer.collect());

    const wantDiagnostics = tick % 6 === 0;
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();

    if (!solver.primed) {
      solver.encodePrime(pass);
    }
    if (!config.paused) {
      for (let i = 0; i < config.substeps; i++) {
        solver.encodeStep(pass);
        solver.encodeDye(pass);
        solver.encodeReduce(pass);
      }
    }
    if (wantDiagnostics) {
      solver.encodeDiagnostics(pass);
    }
    pass.end();

    if (wantDiagnostics) {
      solver.copyDiagnostics(encoder);
    }
    renderer.render(encoder, solver, config);
    device.queue.submit([encoder.finish()]);

    if (wantDiagnostics) {
      const pending = solver;
      solver
        .readDiagnostics()
        .then((result) => {
          if (result && pending === solver) {
            diagnostics = result;
            drawSpectrum(dom.spectrum, result, hover);
            drawZonal(dom.zonal, result);
          }
        })
        .catch(() => {});
    }

    if (diagnostics && !config.paused) {
      solver.simTime += diagnostics.dt * config.substeps;
    }

    tick += 1;
    frames += 1;
    const now = performance.now();
    if (now - lastFpsAt > 500) {
      fps = (frames * 1000) / (now - lastFpsAt);
      frames = 0;
      lastFpsAt = now;
      const d = diagnostics;
      dom.stats.textContent = [
        `${fps.toFixed(0)} fps`,
        `${config.resolution}² spectral / ${config.resolution * config.dyeScale}² dye`,
        d ? `dt ${d.dt.toExponential(2)}` : 'dt —',
        d ? `u_max ${d.maxSpeed.toFixed(2)}` : 'u_max —',
        d ? `E ${d.energy.toFixed(3)}` : 'E —',
        d ? `Z ${d.enstrophy.toFixed(2)}` : 'Z —',
        `t ${solver.simTime.toFixed(1)}`,
      ].join('   ');
    }
  }

  dom.status.textContent = gpu.description;
  dom.status.classList.add('visible');
  setTimeout(() => dom.status.classList.remove('visible'), 2600);
  loop();
}

boot();
