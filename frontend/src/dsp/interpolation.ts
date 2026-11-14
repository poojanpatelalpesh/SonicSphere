/**
 * HRIR interpolation for arbitrary directions – mirrors backend/sonicsphere/hrir_interpolation.py.
 *
 *  nearest              closest measured direction
 *  barycentric_raw      spherical barycentric weights applied to raw HRIR samples
 *  barycentric_aligned  barycentric weights applied to onset-aligned shapes, TOAs interpolated
 *                       separately and re-inserted with a band-limited fractional shift
 *                       (the method selected by the held-out evaluation)
 *
 * Directions inside unmeasured regions (gap triangles) are filled along the cone of
 * confusion (constant lateral angle) between the two rim crossings; clamping to the
 * closest boundary point is the fallback.
 */

import { angularDistanceDeg, cartToSph, quantizeDirection, RAD, sphToCart, type Vec3 } from "./coordinates";
import type { FilterPack } from "./filterPack";
import { FracShifter } from "./fracshift";

export type InterpMethod = "nearest" | "barycentric_aligned" | "barycentric_raw";
export type DirStatus = "measured" | "interpolated" | "gap_fill" | "clamped";

export const INTERP_METHODS: { id: InterpMethod; label: string }[] = [
  { id: "barycentric_aligned", label: "Barycentric · onset-aligned" },
  { id: "barycentric_raw", label: "Barycentric · raw samples" },
  { id: "nearest", label: "Nearest neighbour" },
];

export interface Weights {
  n: number;
  indices: number[];
  weights: number[];
  status: DirStatus;
  triangle: number;
  maxEdgeDeg: number;
  nearestIndex: number;
  nearestAngleDeg: number;
  rendered: Vec3;
  lateralDeg?: number;
  polarLoDeg?: number;
  polarHiDeg?: number;
  t?: number;
  clampAngleDeg?: number;
}

export interface FilterResult {
  key: string;
  az: number;            // quantised direction the filter was computed for
  el: number;
  filter: Float32Array;  // (2, L): left then right
  toa: [number, number];
  w: Weights;
  spectra?: unknown;     // filled lazily by the partitioned convolver
  reversed?: Float64Array; // filled lazily by the direct convolver
}

export class HRIRInterpolator {
  readonly pack: FilterPack;
  method: InterpMethod;
  /** Filter grid (degrees); defaults to the pack value, raised by the worklet in economy mode. */
  quantum: number;
  private readonly shifter: FracShifter;
  private readonly shapeBuf: Float64Array;
  private readonly outBuf: Float64Array;
  private lastTri = -1;
  private readonly cache = new Map<string, FilterResult>();
  readonly cacheLimit: number;
  cacheHits = 0;
  cacheMisses = 0;
  private readonly crossPol: Float64Array;
  private readonly crossI: Int32Array;
  private readonly crossJ: Int32Array;
  private readonly crossA: Float64Array;

  constructor(pack: FilterPack, method: InterpMethod = "barycentric_aligned", cacheLimit = 768) {
    this.pack = pack;
    this.method = method;
    this.quantum = pack.header.directionQuantumDeg;
    this.shifter = new FracShifter(pack.header.fracShiftFft);
    this.shapeBuf = new Float64Array(pack.L);
    this.outBuf = new Float64Array(pack.L);
    this.cacheLimit = cacheLimit;
    const ne = pack.boundaryEdges.length / 2;
    this.crossPol = new Float64Array(2 * ne);
    this.crossI = new Int32Array(2 * ne);
    this.crossJ = new Int32Array(2 * ne);
    this.crossA = new Float64Array(2 * ne);
  }

  setMethod(m: InterpMethod): void {
    if (m !== this.method) {
      this.method = m;
      this.cache.clear();
    }
  }

  clearCache(): void { this.cache.clear(); }

  setQuantum(q: number): void {
    if (q !== this.quantum) { this.quantum = q; this.cache.clear(); }
  }
  get cacheSize(): number { return this.cache.size; }

  // ------------------------------------------------------------- weights
  nearest(p: ArrayLike<number>): { index: number; angle: number } {
    const v = this.pack.vertices;
    let best = -Infinity, bi = 0;
    for (let i = 0, M = this.pack.M; i < M; i++) {
      const d = v[3 * i] * p[0] + v[3 * i + 1] * p[1] + v[3 * i + 2] * p[2];
      if (d > best) { best = d; bi = i; }
    }
    return { index: bi, angle: angularDistanceDeg([v[3 * bi], v[3 * bi + 1], v[3 * bi + 2]], p) };
  }

  /** Triangle maximising the smallest barycentric coordinate (exactly one contains p). */
  locate(p: ArrayLike<number>, w: Float64Array): number {
    const inv = this.pack.triInverse;
    const t0 = this.lastTri;
    if (t0 >= 0) {
      const o = 9 * t0;
      const a = inv[o] * p[0] + inv[o + 1] * p[1] + inv[o + 2] * p[2];
      const b = inv[o + 3] * p[0] + inv[o + 4] * p[1] + inv[o + 5] * p[2];
      const c = inv[o + 6] * p[0] + inv[o + 7] * p[1] + inv[o + 8] * p[2];
      if (Math.min(a, b, c) > 1e-7) { w[0] = a; w[1] = b; w[2] = c; return t0; } // strictly inside: unique
    }
    let best = -Infinity, bt = 0, ba = 0, bb = 0, bc = 0;
    for (let t = 0, T = this.pack.T; t < T; t++) {
      const o = 9 * t;
      const a = inv[o] * p[0] + inv[o + 1] * p[1] + inv[o + 2] * p[2];
      const b = inv[o + 3] * p[0] + inv[o + 4] * p[1] + inv[o + 5] * p[2];
      const c = inv[o + 6] * p[0] + inv[o + 7] * p[1] + inv[o + 8] * p[2];
      const m = a < b ? (a < c ? a : c) : (b < c ? b : c);
      if (m > best) { best = m; bt = t; ba = a; bb = b; bc = c; }
    }
    w[0] = ba; w[1] = bb; w[2] = bc;
    this.lastTri = bt;
    return bt;
  }

  private vert(i: number): Vec3 {
    const v = this.pack.vertices;
    return [v[3 * i], v[3 * i + 1], v[3 * i + 2]];
  }

  /** Crossings of the constant-lateral circle through p with the boundary edges. */
  lateralCrossings(p: ArrayLike<number>): number {
    const s = p[1];
    const be = this.pack.boundaryEdges;
    let n = 0;
    for (let e = 0; e < be.length / 2; e++) {
      const i = be[2 * e], j = be[2 * e + 1];
      const a = this.vert(i), b = this.vert(j);
      const cosT = Math.max(-1, Math.min(1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]));
      const T = Math.acos(cosT);
      let c0 = b[0] - cosT * a[0], c1 = b[1] - cosT * a[1], c2 = b[2] - cosT * a[2];
      const cn = Math.hypot(c0, c1, c2);
      if (cn < 1e-12) continue;
      c0 /= cn; c1 /= cn; c2 /= cn;
      const R = Math.hypot(a[1], c1);
      if (R < 1e-12 || Math.abs(s) > R) continue;
      const phi = Math.atan2(c1, a[1]);
      const d = Math.acos(Math.max(-1, Math.min(1, s / R)));
      for (const t0 of [phi - d, phi + d]) {
        // Python: (t + pi) % (2 pi) - pi  with a floored modulo
        const twoPi = 2 * Math.PI;
        let t = (t0 + Math.PI) % twoPi;
        if (t < 0) t += twoPi;
        t -= Math.PI;
        if (t >= -1e-12 && t <= T + 1e-12) {
          const qx = a[0] * Math.cos(t) + c0 * Math.sin(t);
          const qz = a[2] * Math.cos(t) + c2 * Math.sin(t);
          let kb = Math.sin(t) / Math.sin(T);
          let ka = Math.cos(t) - kb * cosT;
          ka = Math.max(ka, 0); kb = Math.max(kb, 0);
          this.crossPol[n] = Math.atan2(qz, qx) * RAD;
          this.crossI[n] = i;
          this.crossJ[n] = j;
          this.crossA[n] = ka + kb > 0 ? kb / (ka + kb) : 0;
          n++;
        }
      }
    }
    return n;
  }

  private gapFill(p: ArrayLike<number>, out: Weights): boolean {
    const n = this.lateralCrossings(p);
    if (n < 2) return false;
    const beta = Math.atan2(p[2], p[0]) * RAD;
    const mod = (x: number) => { const r = x % 360; return r < 0 ? r + 360 : r; };
    let dd = Infinity, klo = -1, du = Infinity, khi = -1;
    for (let k = 0; k < n; k++) {
      const down = mod(beta - this.crossPol[k]);
      const up = mod(this.crossPol[k] - beta);
      if (down < dd) { dd = down; klo = k; }
      if (up < du) { du = up; khi = k; }
    }
    if (klo === khi || dd + du <= 1e-9) return false;
    const t = dd / (dd + du);
    const acc = new Map<number, number>();
    const add = (k: number, wt: number) => {
      const i = this.crossI[k], j = this.crossJ[k], al = this.crossA[k];
      acc.set(i, (acc.get(i) ?? 0) + wt * (1 - al));
      acc.set(j, (acc.get(j) ?? 0) + wt * al);
    };
    add(klo, 1 - t);
    add(khi, t);
    const idx = [...acc.keys()].sort((x, y) => x - y).filter((k) => (acc.get(k) ?? 0) > 0);
    let sum = 0;
    for (const k of idx) sum += acc.get(k)!;
    out.n = idx.length;
    out.indices = idx;
    out.weights = idx.map((k) => acc.get(k)! / sum);
    out.status = "gap_fill";
    out.rendered = [p[0], p[1], p[2]];
    out.lateralDeg = Math.asin(Math.max(-1, Math.min(1, p[1]))) * RAD;
    out.polarLoDeg = this.crossPol[klo];
    out.polarHiDeg = this.crossPol[khi];
    out.t = t;
    return true;
  }

  private clamp(p: ArrayLike<number>, out: Weights): void {
    const be = this.pack.boundaryEdges;
    let best = -Infinity, bq: Vec3 = [1, 0, 0], bi = 0, bj = 0;
    for (let e = 0; e < be.length / 2; e++) {
      const i = be[2 * e], j = be[2 * e + 1];
      const a = this.vert(i), b = this.vert(j);
      let n0 = a[1] * b[2] - a[2] * b[1], n1 = a[2] * b[0] - a[0] * b[2], n2 = a[0] * b[1] - a[1] * b[0];
      const nn = Math.hypot(n0, n1, n2);
      n0 /= nn; n1 /= nn; n2 /= nn;
      const pn = p[0] * n0 + p[1] * n1 + p[2] * n2;
      let q0 = p[0] - pn * n0, q1 = p[1] - pn * n1, q2 = p[2] - pn * n2;
      const qn = Math.hypot(q0, q1, q2);
      const qs = qn > 1e-12 ? qn : 1;
      q0 /= qs; q1 /= qs; q2 /= qs;
      // inside: (a x q).n >= 0 and (q x b).n >= 0
      const aq = (a[1] * q2 - a[2] * q1) * n0 + (a[2] * q0 - a[0] * q2) * n1 + (a[0] * q1 - a[1] * q0) * n2;
      const qb = (q1 * b[2] - q2 * b[1]) * n0 + (q2 * b[0] - q0 * b[2]) * n1 + (q0 * b[1] - q1 * b[0]) * n2;
      let cand: Vec3;
      if (aq >= 0 && qb >= 0 && qn > 1e-12) cand = [q0, q1, q2];
      else {
        const da = a[0] * p[0] + a[1] * p[1] + a[2] * p[2], db = b[0] * p[0] + b[1] * p[1] + b[2] * p[2];
        cand = da >= db ? a : b;
      }
      const score = cand[0] * p[0] + cand[1] * p[1] + cand[2] * p[2];
      if (score > best) { best = score; bq = cand; bi = i; bj = j; }
    }
    const vi = this.vert(bi), vj = this.vert(bj);
    const g00 = vi[0] * vi[0] + vi[1] * vi[1] + vi[2] * vi[2];
    const g01 = vi[0] * vj[0] + vi[1] * vj[1] + vi[2] * vj[2];
    const g11 = vj[0] * vj[0] + vj[1] * vj[1] + vj[2] * vj[2];
    const r0 = vi[0] * bq[0] + vi[1] * bq[1] + vi[2] * bq[2];
    const r1 = vj[0] * bq[0] + vj[1] * bq[1] + vj[2] * bq[2];
    const det = g00 * g11 - g01 * g01;
    let c0 = (g11 * r0 - g01 * r1) / det, c1 = (g00 * r1 - g01 * r0) / det;
    c0 = Math.max(c0, 0); c1 = Math.max(c1, 0);
    const s = c0 + c1;
    const alpha = s > 0 ? c1 / s : 0;
    out.n = 2;
    out.indices = [bi, bj];
    out.weights = [1 - alpha, alpha];
    out.status = "clamped";
    out.rendered = bq;
    out.clampAngleDeg = angularDistanceDeg(bq, p);
  }

  weights(dir: ArrayLike<number>): Weights {
    const nrm = Math.hypot(dir[0], dir[1], dir[2]) || 1;
    const p: Vec3 = [dir[0] / nrm, dir[1] / nrm, dir[2] / nrm];
    const { index: ni, angle: nang } = this.nearest(p);
    const base: Weights = { n: 1, indices: [ni], weights: [1], status: "measured", triangle: -1, maxEdgeDeg: 0,
                            nearestIndex: ni, nearestAngleDeg: nang, rendered: this.vert(ni) };
    if (this.method === "nearest" || nang <= this.pack.header.measuredSnapDeg) return base;
    const bw = new Float64Array(3);
    const t = this.locate(p, bw);
    base.triangle = t;
    base.maxEdgeDeg = this.pack.triMaxEdge[t];
    if (this.pack.triGap[t]) {
      if (this.gapFill(p, base)) return base;
      this.clamp(p, base);
      return base;
    }
    let a = Math.max(bw[0], 0), b = Math.max(bw[1], 0), c = Math.max(bw[2], 0);
    const s = a + b + c;
    a /= s; b /= s; c /= s;
    const tri = this.pack.triangles;
    base.n = 3;
    base.indices = [tri[3 * t], tri[3 * t + 1], tri[3 * t + 2]];
    base.weights = [a, b, c];
    base.status = "interpolated";
    base.rendered = p;
    return base;
  }

  // ------------------------------------------------------------- filters
  /** Filter (2·L, left then right) for a direction, computed exactly (not quantised). */
  compute(dir: ArrayLike<number>, out: Float32Array = new Float32Array(2 * this.pack.L)): { w: Weights; toa: [number, number]; filter: Float32Array } {
    const w = this.weights(dir);
    const { L, measured, aligned, toa } = this.pack;
    const margin = this.pack.header.margin;
    const toaOut: [number, number] = [0, 0];
    if (w.status === "measured") {
      const i = w.indices[0];
      out.set(measured.subarray(i * 2 * L, (i + 1) * 2 * L));
      toaOut[0] = toa[2 * i]; toaOut[1] = toa[2 * i + 1];
      return { w, toa: toaOut, filter: out };
    }
    for (let ear = 0; ear < 2; ear++) {
      let tsum = 0;
      for (let k = 0; k < w.n; k++) tsum += w.weights[k] * toa[2 * w.indices[k] + ear];
      toaOut[ear] = tsum;
      const src = this.method === "barycentric_raw" ? measured : aligned;
      const shape = this.shapeBuf;
      shape.fill(0);
      for (let k = 0; k < w.n; k++) {
        const off = (w.indices[k] * 2 + ear) * L;
        const wk = w.weights[k];
        for (let n = 0; n < L; n++) shape[n] += wk * src[off + n];
      }
      if (this.method === "barycentric_raw") {
        for (let n = 0; n < L; n++) out[ear * L + n] = shape[n];
      } else {
        this.shifter.shift(shape, tsum - margin, this.outBuf, L);
        for (let n = 0; n < L; n++) out[ear * L + n] = this.outBuf[n];
      }
    }
    return { w, toa: toaOut, filter: out };
  }

  /** Cached filter on the quantised direction grid (what the real-time engine uses). */
  query(dir: ArrayLike<number>): FilterResult {
    const { az, el } = cartToSph(dir);
    const [azQ, elQ] = quantizeDirection(az, el, this.quantum);
    const key = `${azQ}|${elQ}`;
    const hit = this.cache.get(key);
    if (hit) {
      this.cacheHits++;
      this.cache.delete(key);
      this.cache.set(key, hit);  // LRU: move to the most recent position
      return hit;
    }
    this.cacheMisses++;
    const r = this.compute(sphToCart(azQ, elQ));
    const res: FilterResult = { key, az: azQ, el: elQ, filter: r.filter, toa: r.toa, w: r.w };
    this.cache.set(key, res);
    if (this.cache.size > this.cacheLimit) this.cache.delete(this.cache.keys().next().value as string);
    return res;
  }
}
