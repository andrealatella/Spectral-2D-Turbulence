import { MAX_SPLATS } from '../gpu/solver.js';

const PALETTE = [
  [1.0, 0.42, 0.28],
  [0.34, 0.78, 1.0],
  [1.0, 0.82, 0.32],
  [0.62, 0.48, 1.0],
  [0.36, 0.93, 0.66],
  [1.0, 0.5, 0.78],
];

export class Pointer {
  constructor(canvas) {
    this.canvas = canvas;
    this.pending = [];
    this.active = false;
    this.last = null;
    this.stroke = 0;
    this.radius = 0.045;
    this.strength = 400;
    this.dyeAmount = 1.0;

    canvas.addEventListener('pointerdown', (e) => this.onDown(e));
    canvas.addEventListener('pointermove', (e) => this.onMove(e));
    for (const type of ['pointerup', 'pointercancel', 'pointerleave']) {
      canvas.addEventListener(type, () => this.onUp());
    }
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  locate(e) {
    const r = this.canvas.getBoundingClientRect();
    return {
      x: (e.clientX - r.left) / r.width,
      y: 1 - (e.clientY - r.top) / r.height,
    };
  }

  onDown(e) {
    this.canvas.setPointerCapture(e.pointerId);
    this.active = true;
    this.stroke = (this.stroke + 1) % PALETTE.length;
    this.last = this.locate(e);
  }

  onMove(e) {
    if (!this.active) {
      return;
    }
    const p = this.locate(e);
    const dx = p.x - this.last.x;
    const dy = p.y - this.last.y;
    this.last = p;
    if (dx === 0 && dy === 0) {
      return;
    }
    if (this.pending.length < MAX_SPLATS) {
      this.pending.push({
        x: p.x,
        y: p.y,
        dx,
        dy,
        color: PALETTE[this.stroke],
        radius: this.radius,
        amp: this.strength,
        dyeAmp: this.dyeAmount * Math.min(1, Math.hypot(dx, dy) * 40),
      });
    }
  }

  onUp() {
    this.active = false;
    this.last = null;
  }

  collect() {
    const out = this.pending;
    this.pending = [];
    return out;
  }
}
