const INK = '#ffffff';
const INK_DIM = '#c3c2b7';
const GRID = 'rgba(195, 194, 183, 0.14)';
const SERIES = '#3987e5';
const EAST = '#e66767';
const WEST = '#3987e5';
const NEUTRAL = '#383835';

function setup(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.textBaseline = 'middle';
  return { ctx, w, h };
}

function frame(ctx, w, h, pad) {
  ctx.strokeStyle = GRID;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t);
  ctx.lineTo(pad.l, h - pad.b);
  ctx.lineTo(w - pad.r, h - pad.b);
  ctx.stroke();
}

export function drawSpectrum(canvas, data, hover) {
  const { ctx, w, h } = setup(canvas);
  const pad = { l: 34, r: 10, t: 16, b: 20 };
  if (!data || !data.spectrum) {
    return;
  }

  const spec = data.spectrum;
  const kMax = spec.length - 1;
  let peak = 0;
  for (let k = 1; k <= kMax; k++) {
    peak = Math.max(peak, spec[k]);
  }
  if (peak <= 0) {
    frame(ctx, w, h, pad);
    return;
  }

  const hi = Math.ceil(Math.log10(peak)) + 0.5;
  const lo = hi - 7;
  const x = (k) => pad.l + ((Math.log10(k) - 0) / Math.log10(kMax)) * (w - pad.l - pad.r);
  const y = (v) => h - pad.b - ((Math.log10(v) - lo) / (hi - lo)) * (h - pad.t - pad.b);

  ctx.strokeStyle = GRID;
  ctx.lineWidth = 1;
  for (let d = Math.ceil(lo); d <= hi; d += 1) {
    const py = y(10 ** d);
    ctx.beginPath();
    ctx.moveTo(pad.l, py);
    ctx.lineTo(w - pad.r, py);
    ctx.stroke();
  }
  let lastLabelX = -Infinity;
  for (let dec = 1; dec <= kMax; dec *= 10) {
    for (const mul of [1, 2, 5]) {
      const k = dec * mul;
      if (k < 1 || k > kMax) {
        continue;
      }
      const px = x(k);
      ctx.beginPath();
      ctx.moveTo(px, pad.t);
      ctx.lineTo(px, h - pad.b);
      ctx.stroke();
      const width = ctx.measureText(String(k)).width;
      if (px - lastLabelX > width * 0.5 + 9) {
        ctx.fillStyle = INK_DIM;
        ctx.textAlign = 'center';
        ctx.fillText(String(k), px, h - pad.b + 9);
        lastLabelX = px + width * 0.5;
      }
    }
  }

  let anchorK = 1;
  for (let k = 1; k <= kMax; k++) {
    if (spec[k] > spec[anchorK]) {
      anchorK = k;
    }
  }
  ctx.setLineDash([4, 4]);
  ctx.lineWidth = 1;
  ctx.strokeStyle = INK_DIM;
  for (const [slope, label] of [
    [-3, 'k⁻³'],
    [-5 / 3, 'k⁻⁵ᐟ³'],
  ]) {
    const k0 = Math.max(anchorK, 2);
    const e0 = spec[k0] * 3;
    const at = (k) => e0 * (k / k0) ** slope;
    const floor = 10 ** lo;
    let kEnd = kMax;
    while (kEnd > k0 * 1.5 && at(kEnd) < floor) {
      kEnd = Math.floor(kEnd * 0.92);
    }
    if (kEnd <= k0 * 1.5) {
      continue;
    }
    ctx.globalAlpha = 0.45;
    ctx.beginPath();
    ctx.moveTo(x(k0), y(e0));
    ctx.lineTo(x(kEnd), y(at(kEnd)));
    ctx.stroke();

    const kl = k0 * (kEnd / k0) ** 0.66;
    ctx.globalAlpha = 0.9;
    ctx.fillStyle = INK_DIM;
    ctx.textAlign = 'center';
    ctx.fillText(label, x(kl), y(at(kl)) - 8);
  }
  ctx.setLineDash([]);
  ctx.globalAlpha = 1;

  ctx.strokeStyle = SERIES;
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  let started = false;
  for (let k = 1; k <= kMax; k++) {
    const v = spec[k];
    if (v <= 10 ** lo) {
      started = false;
      continue;
    }
    const px = x(k);
    const py = y(v);
    if (started) {
      ctx.lineTo(px, py);
    } else {
      ctx.moveTo(px, py);
      started = true;
    }
  }
  ctx.stroke();

  const floor = 10 ** lo;
  ctx.fillStyle = SERIES;
  for (let k = 1; k <= kMax; k++) {
    if (spec[k] <= floor) {
      continue;
    }
    const hasPrev = k > 1 && spec[k - 1] > floor;
    const hasNext = k < kMax && spec[k + 1] > floor;
    if (!hasPrev && !hasNext) {
      ctx.beginPath();
      ctx.arc(x(k), y(spec[k]), 2.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  ctx.fillStyle = INK_DIM;
  ctx.textAlign = 'right';
  for (let d = Math.ceil(lo); d <= hi; d += 2) {
    ctx.fillText(`1e${d}`, pad.l - 4, y(10 ** d));
  }

  ctx.fillStyle = INK;
  ctx.textAlign = 'left';
  ctx.fillText('E(k)', pad.l + 2, pad.t - 8);

  if (hover && hover.x > pad.l && hover.x < w - pad.r) {
    const frac = (hover.x - pad.l) / (w - pad.l - pad.r);
    const k = Math.round(10 ** (frac * Math.log10(kMax)));
    if (k >= 1 && k <= kMax && spec[k] > 0) {
      ctx.strokeStyle = 'rgba(255,255,255,0.35)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x(k), pad.t);
      ctx.lineTo(x(k), h - pad.b);
      ctx.stroke();
      ctx.fillStyle = INK;
      ctx.textAlign = 'right';
      ctx.fillText(`k=${k}  ${spec[k].toExponential(2)}`, w - pad.r - 2, pad.t - 8);
    }
  }

  frame(ctx, w, h, pad);
}

export function drawZonal(canvas, data) {
  const { ctx, w, h } = setup(canvas);
  const pad = { l: 10, r: 10, t: 16, b: 16 };
  if (!data || !data.zonal || !data.zonal.length) {
    return;
  }

  const u = data.zonal;
  let amp = 1e-6;
  for (let i = 0; i < u.length; i++) {
    amp = Math.max(amp, Math.abs(u[i]));
  }

  const midX = (pad.l + (w - pad.r)) / 2;
  const halfW = (w - pad.l - pad.r) / 2;
  const y = (i) => h - pad.b - (i / (u.length - 1)) * (h - pad.t - pad.b);
  const x = (v) => midX + (v / amp) * halfW * 0.92;

  ctx.fillStyle = EAST;
  ctx.globalAlpha = 0.75;
  ctx.beginPath();
  ctx.moveTo(midX, y(0));
  for (let i = 0; i < u.length; i++) {
    ctx.lineTo(x(Math.max(u[i], 0)), y(i));
  }
  ctx.lineTo(midX, y(u.length - 1));
  ctx.closePath();
  ctx.fill();

  ctx.fillStyle = WEST;
  ctx.beginPath();
  ctx.moveTo(midX, y(0));
  for (let i = 0; i < u.length; i++) {
    ctx.lineTo(x(Math.min(u[i], 0)), y(i));
  }
  ctx.lineTo(midX, y(u.length - 1));
  ctx.closePath();
  ctx.fill();
  ctx.globalAlpha = 1;

  ctx.strokeStyle = NEUTRAL;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(midX, pad.t);
  ctx.lineTo(midX, h - pad.b);
  ctx.stroke();

  ctx.fillStyle = INK;
  ctx.textAlign = 'left';
  ctx.fillText('u(y)', pad.l, pad.t - 8);
  ctx.fillStyle = INK_DIM;
  ctx.textAlign = 'right';
  ctx.fillText(`±${amp.toFixed(2)}`, w - pad.r, pad.t - 8);
}
