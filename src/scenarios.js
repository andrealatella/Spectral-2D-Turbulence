const TWO_PI = Math.PI * 2;

function grid(n, fn) {
  const out = new Float32Array(n * n);
  const h = TWO_PI / n;
  for (let row = 0; row < n; row++) {
    const y = row * h;
    for (let col = 0; col < n; col++) {
      out[row * n + col] = fn(col * h, y);
    }
  }
  return out;
}

function sech2(z) {
  const c = Math.cosh(z);
  return 1 / (c * c);
}

function wrapDelta(a, b) {
  let d = a - b;
  if (d > Math.PI) {
    d -= TWO_PI;
  } else if (d < -Math.PI) {
    d += TWO_PI;
  }
  return d;
}

function blobs(n, list) {
  return grid(n, (x, y) => {
    let acc = 0;
    for (const b of list) {
      const dx = wrapDelta(x, b.x);
      const dy = wrapDelta(y, b.y);
      const r2 = dx * dx + dy * dy;
      const s2 = b.a * b.a;
      if (r2 < 25 * s2) {
        acc += b.amp * Math.exp(-r2 / s2);
      }
    }
    return acc;
  });
}

function ring(n, count, radius, a, amp, jitter) {
  const list = [];
  const mode = Math.max(2, Math.floor(count / 2));
  for (let j = 0; j < count; j++) {
    const th = (TWO_PI * j) / count;
    const r = radius * (1 + jitter * Math.cos((TWO_PI * mode * j) / count));
    list.push({ x: Math.PI + r * Math.cos(th), y: Math.PI + r * Math.sin(th), a, amp });
  }
  return blobs(n, list);
}

export const SCENARIOS = {
  'shear layer': {
    config: {
      substeps: 2,
      viscosity: 0.5,
      hyperP: 4,
      beta: 0,
      drag: 0,
      forceAmp: 0,
      mode: 'vorticity',
      vortScale: 0.16,
      exposure: 1.3,
    },
    build(n) {
      const delta = Math.max(TWO_PI / 30, (6 * TWO_PI) / n);
      const eps = 0.05;
      return grid(n, (x, y) => {
        const lower = y <= Math.PI;
        const z = lower ? (y - Math.PI / 2) / delta : (1.5 * Math.PI - y) / delta;
        const dudy = ((lower ? 1 : -1) * sech2(z)) / delta;
        return eps * Math.cos(x) - dudy;
      });
    },
  },

  'taylor-green': {
    config: {
      substeps: 4,
      viscosity: 90,
      hyperP: 1,
      beta: 0,
      drag: 0,
      forceAmp: 0,
      mode: 'vorticity',
      vortScale: 0.16,
      exposure: 1.3,
    },
    build(n) {
      const k = 2;
      return grid(n, (x, y) => 6 * Math.cos(k * x) * Math.cos(k * y));
    },
  },

  dipole: {
    config: {
      substeps: 4,
      viscosity: 0.3,
      hyperP: 4,
      beta: 0,
      drag: 0,
      forceAmp: 0,
      mode: 'dye over vorticity',
      vortScale: 0.022,
      exposure: 1.3,
    },
    build(n) {
      const gamma = 12;
      const a = 0.3;
      const d = 0.9;
      const amp = gamma / (Math.PI * a * a);
      const cx = Math.PI * 0.6;
      return blobs(n, [
        { x: cx, y: Math.PI - d / 2, a, amp },
        { x: cx, y: Math.PI + d / 2, a, amp: -amp },
      ]);
    },
  },

  merger: {
    config: {
      substeps: 8,
      viscosity: 0.3,
      hyperP: 4,
      beta: 0,
      drag: 0,
      forceAmp: 0,
      mode: 'vorticity',
      vortScale: 0.032,
      exposure: 1.3,
    },
    build(n, cfg) {
      const a = 0.22;
      const d = cfg.mergerSep;
      return blobs(n, [
        { x: Math.PI - d / 2, y: Math.PI, a, amp: 30 },
        { x: Math.PI + d / 2, y: Math.PI, a, amp: 30 },
      ]);
    },
  },

  polygon: {
    config: {
      substeps: 8,
      viscosity: 0.22,
      hyperP: 4,
      beta: 0,
      drag: 0,
      forceAmp: 0,
      mode: 'vorticity',
      vortScale: 0.026,
      exposure: 1.3,
    },
    build(n, cfg) {
      return ring(n, Math.round(cfg.polygonN), 1.25, 0.19, 34, 0.02);
    },
  },
};

export const SCENARIO_NAMES = Object.keys(SCENARIOS);
