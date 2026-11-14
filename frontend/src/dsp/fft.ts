/**
 * Radix-2 FFTs for the real-time engine (no allocation after construction).
 *
 * ComplexFFT: iterative in-place Cooley–Tukey on separate re/im arrays.
 * RealFFT:    an N-point real FFT computed with one N/2-point complex FFT
 *             (even/odd packing), returning bins 0..N/2 – the same convention as
 *             numpy.fft.rfft / irfft.
 */

export class ComplexFFT {
  readonly n: number;
  private readonly rev: Uint32Array;
  private readonly cos: Float64Array;
  private readonly sin: Float64Array;

  constructor(n: number) {
    if (n < 1 || (n & (n - 1)) !== 0) throw new Error(`FFT size must be a power of two (got ${n})`);
    this.n = n;
    const bits = Math.round(Math.log2(n));
    this.rev = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
      this.rev[i] = r;
    }
    this.cos = new Float64Array(n / 2);
    this.sin = new Float64Array(n / 2);
    for (let k = 0; k < n / 2; k++) {
      this.cos[k] = Math.cos((2 * Math.PI * k) / n);
      this.sin[k] = Math.sin((2 * Math.PI * k) / n);
    }
  }

  /** In-place transform. inverse=true computes the unscaled inverse (caller scales by 1/n). */
  transform(re: Float64Array, im: Float64Array, inverse = false): void {
    const n = this.n;
    const rev = this.rev;
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    const sgn = inverse ? 1 : -1;
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1;
      const step = n / size;
      for (let start = 0; start < n; start += size) {
        for (let k = 0; k < half; k++) {
          const wr = this.cos[k * step];
          const wi = sgn * this.sin[k * step];
          const a = start + k;
          const b = a + half;
          const xr = re[b] * wr - im[b] * wi;
          const xi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - xr;
          im[b] = im[a] - xi;
          re[a] += xr;
          im[a] += xi;
        }
      }
    }
  }
}

export class RealFFT {
  readonly n: number;
  readonly bins: number;
  private readonly half: ComplexFFT;
  private readonly zr: Float64Array;
  private readonly zi: Float64Array;
  private readonly wr: Float64Array;
  private readonly wi: Float64Array;

  constructor(n: number) {
    if (n < 2 || (n & (n - 1)) !== 0) throw new Error(`RealFFT size must be a power of two >= 2 (got ${n})`);
    this.n = n;
    this.bins = n / 2 + 1;
    this.half = new ComplexFFT(n / 2);
    this.zr = new Float64Array(n / 2);
    this.zi = new Float64Array(n / 2);
    this.wr = new Float64Array(n / 2);
    this.wi = new Float64Array(n / 2);
    for (let k = 0; k < n / 2; k++) {
      this.wr[k] = Math.cos((2 * Math.PI * k) / n);
      this.wi[k] = -Math.sin((2 * Math.PI * k) / n);
    }
  }

  /** X[k] = sum_n x[n] e^{-j 2 pi k n / N}, k = 0..N/2. `x` may be shorter than N (zero-padded). */
  forward(x: ArrayLike<number>, outRe: Float64Array, outIm: Float64Array, xLength = x.length): void {
    const m = this.n / 2;
    const zr = this.zr, zi = this.zi;
    for (let i = 0; i < m; i++) {
      const a = 2 * i, b = a + 1;
      zr[i] = a < xLength ? x[a] : 0;
      zi[i] = b < xLength ? x[b] : 0;
    }
    this.half.transform(zr, zi, false);
    // E[k] = (Z[k] + conj(Z[m-k])) / 2,  O[k] = (Z[k] - conj(Z[m-k])) / (2j),  X[k] = E[k] + W^k O[k]
    for (let k = 0; k <= m; k++) {
      const k1 = k % m;
      const k2 = (m - k) % m;
      const ar = zr[k1], ai = zi[k1];
      const br = zr[k2], bi = -zi[k2];
      const er = 0.5 * (ar + br), ei = 0.5 * (ai + bi);
      // (a - b) / (2j) = -j (a - b) / 2
      const dr = 0.5 * (ai - bi), di = -0.5 * (ar - br);
      let wr: number, wi: number;
      if (k === m) { wr = -1; wi = 0; } else { wr = this.wr[k]; wi = this.wi[k]; }
      outRe[k] = er + (wr * dr - wi * di);
      outIm[k] = ei + (wr * di + wi * dr);
    }
  }

  /** Inverse of forward (scaled by 1/N), like numpy.fft.irfft; Im of bins 0 and N/2 is ignored. */
  inverse(inRe: Float64Array, inIm: Float64Array, out: Float64Array | Float32Array): void {
    const m = this.n / 2;
    const zr = this.zr, zi = this.zi;
    for (let k = 0; k < m; k++) {
      // E[k] = (X[k] + conj(X[m-k])) / 2,  O[k] = (X[k] - conj(X[m-k])) * W^{-k} / 2,  Z = E + j O
      const xr = inRe[k], xi = k === 0 ? 0 : inIm[k];
      const yr = inRe[m - k], yi = m - k === m || m - k === 0 ? 0 : -inIm[m - k];
      const er = 0.5 * (xr + yr), ei = 0.5 * (xi + yi);
      const dr = 0.5 * (xr - yr), di = 0.5 * (xi - yi);
      const wr = this.wr[k], wi = -this.wi[k]; // W^{-k}
      const or = dr * wr - di * wi, oi = dr * wi + di * wr;
      zr[k] = er - oi;
      zi[k] = ei + or;
    }
    this.half.transform(zr, zi, true);
    const s = 1 / m;
    for (let i = 0; i < m; i++) {
      out[2 * i] = zr[i] * s;
      out[2 * i + 1] = zi[i] * s;
    }
  }
}
