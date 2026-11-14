/**
 * Trajectories p_s(t) – mirrors backend/sonicsphere/trajectory.py.
 * Centripetal Catmull–Rom spline (α = 0.5) through control points, resampled uniformly
 * by arc length; playback position = phase ∈ [0, 1] along the resampled polyline.
 */

import type { Vec3 } from "./coordinates";

export const ALPHA = 0.5;
export const SEGMENT_SUBDIVISIONS = 32;
export const DEFAULT_SAMPLES = 512;

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const len = (a: Vec3) => Math.hypot(a[0], a[1], a[2]);

function segment(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, n: number, out: Vec3[]): void {
  const knot = (ti: number, a: Vec3, b: Vec3) => ti + Math.max(Math.pow(len(sub(b, a)), ALPHA), 1e-4);
  const t0 = 0, t1 = knot(t0, p0, p1), t2 = knot(t1, p1, p2), t3 = knot(t2, p2, p3);
  for (let s = 0; s < n; s++) {
    const t = t1 + ((t2 - t1) * s) / n;
    const q: Vec3 = [0, 0, 0];
    for (let c = 0; c < 3; c++) {
      const a1 = ((t1 - t) / (t1 - t0)) * p0[c] + ((t - t0) / (t1 - t0)) * p1[c];
      const a2 = ((t2 - t) / (t2 - t1)) * p1[c] + ((t - t1) / (t2 - t1)) * p2[c];
      const a3 = ((t3 - t) / (t3 - t2)) * p2[c] + ((t - t2) / (t3 - t2)) * p3[c];
      const b1 = ((t2 - t) / (t2 - t0)) * a1 + ((t - t0) / (t2 - t0)) * a2;
      const b2 = ((t3 - t) / (t3 - t1)) * a2 + ((t - t1) / (t3 - t1)) * a3;
      q[c] = ((t2 - t) / (t2 - t1)) * b1 + ((t - t1) / (t2 - t1)) * b2;
    }
    out.push(q);
  }
}

export function splinePolyline(points: Vec3[], closed: boolean): Vec3[] {
  const k = points.length;
  if (k === 0) throw new Error("trajectory needs at least one point");
  if (k === 1) return [points[0].slice() as Vec3];
  if (k === 2 && !closed) {
    const out: Vec3[] = [];
    for (let i = 0; i <= SEGMENT_SUBDIVISIONS; i++) {
      const u = i / SEGMENT_SUBDIVISIONS;
      out.push([0, 1, 2].map((c) => points[0][c] + u * (points[1][c] - points[0][c])) as Vec3);
    }
    return out;
  }
  const out: Vec3[] = [];
  const nSeg = closed ? k : k - 1;
  const ext = (a: Vec3, b: Vec3): Vec3 => [2 * a[0] - b[0], 2 * a[1] - b[1], 2 * a[2] - b[2]];
  for (let i = 0; i < nSeg; i++) {
    let q0: Vec3, q1: Vec3, q2: Vec3, q3: Vec3;
    if (closed) {
      q0 = points[(i - 1 + k) % k]; q1 = points[i]; q2 = points[(i + 1) % k]; q3 = points[(i + 2) % k];
    } else {
      q1 = points[i]; q2 = points[i + 1];
      q0 = i > 0 ? points[i - 1] : ext(points[0], points[1]);
      q3 = i + 2 < k ? points[i + 2] : ext(points[k - 1], points[k - 2]);
    }
    segment(q0, q1, q2, q3, SEGMENT_SUBDIVISIONS, out);
  }
  out.push((closed ? points[0] : points[k - 1]).slice() as Vec3);
  return out;
}

export function resampleByArclength(poly: Vec3[], n = DEFAULT_SAMPLES): Float64Array {
  const out = new Float64Array(3 * n);
  if (poly.length === 1) {
    for (let i = 0; i < n; i++) out.set(poly[0], 3 * i);
    return out;
  }
  const seg: number[] = [];
  const cum: number[] = [0];
  for (let i = 1; i < poly.length; i++) {
    const d = len(sub(poly[i], poly[i - 1]));
    seg.push(d);
    cum.push(cum[i - 1] + d);
  }
  const total = cum[cum.length - 1];
  if (total <= 1e-12) {
    for (let i = 0; i < n; i++) out.set(poly[0], 3 * i);
    return out;
  }
  let j = 0;
  for (let i = 0; i < n; i++) {
    // numpy.linspace(0, total, n): start + i*step, last element exactly `total`
    const s = i === n - 1 ? total : i * (total / (n - 1));
    while (j < seg.length - 1 && cum[j + 1] < s) j++;
    const span = seg[j];
    let u = span <= 1e-12 ? 0 : (s - cum[j]) / span;
    u = Math.min(1, Math.max(0, u));
    for (let c = 0; c < 3; c++) out[3 * i + c] = poly[j][c] + u * (poly[j + 1][c] - poly[j][c]);
  }
  return out;
}

export class Trajectory {
  readonly samples: Float64Array; // (S, 3) flattened
  readonly count: number;
  durationS: number;
  loop: boolean;

  constructor(samples: Float64Array, durationS: number, loop: boolean) {
    if (!(durationS > 0)) throw new Error("duration must be positive");
    this.samples = samples;
    this.count = samples.length / 3;
    this.durationS = durationS;
    this.loop = loop;
  }

  static fromPoints(points: Vec3[], durationS: number, loop = true, closed: boolean = loop, n = DEFAULT_SAMPLES): Trajectory {
    return new Trajectory(resampleByArclength(splinePolyline(points, closed), n), durationS, loop);
  }

  phaseAt(tS: number): number {
    const ph = tS / this.durationS;
    if (this.loop) return ph - Math.floor(ph);
    return Math.min(1, Math.max(0, ph));
  }

  positionAtPhase(phase: number, out: Vec3 = [0, 0, 0]): Vec3 {
    const s = this.samples, S = this.count;
    const u = Math.min(1, Math.max(0, phase)) * (S - 1);
    const i = Math.floor(u);
    if (i >= S - 1) { out[0] = s[3 * (S - 1)]; out[1] = s[3 * (S - 1) + 1]; out[2] = s[3 * (S - 1) + 2]; return out; }
    const f = u - i;
    for (let c = 0; c < 3; c++) out[c] = s[3 * i + c] + f * (s[3 * (i + 1) + c] - s[3 * i + c]);
    return out;
  }

  positionAt(tS: number, out?: Vec3): Vec3 {
    return this.positionAtPhase(this.phaseAt(tS), out);
  }

  length(): number {
    let L = 0;
    for (let i = 1; i < this.count; i++) {
      L += Math.hypot(this.samples[3 * i] - this.samples[3 * i - 3], this.samples[3 * i + 1] - this.samples[3 * i - 2],
                      this.samples[3 * i + 2] - this.samples[3 * i - 1]);
    }
    return L;
  }
}
