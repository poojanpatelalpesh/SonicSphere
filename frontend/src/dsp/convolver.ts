/**
 * Streaming block convolvers for the real-time engine – mirror backend/sonicsphere/convolution.py.
 *
 * DirectConvolver       y[n] = Σ_k h[k] x[n−k] with a persistent input history (O(L) per sample)
 * PartitionedConvolver  uniformly-partitioned overlap-save (UPOLS): FFT size 2B, P = L/B
 *                       partitions, frequency-domain delay line; latency = one block.
 *
 * Both keep the input history separate from the filter, so the outputs of two filters
 * for the same block can be crossfaded when the source direction changes.
 */

import { RealFFT } from "./fft";

export type ConvMethod = "direct" | "partitioned";

export interface FilterSpectra { re: Float64Array; im: Float64Array } // (2 ears · P partitions · (B+1))

export class DirectConvolver {
  readonly L: number;
  readonly B: number;
  private readonly hist: Float64Array; // length L-1+B, oldest first

  constructor(L: number, B: number) {
    this.L = L;
    this.B = B;
    this.hist = new Float64Array(L - 1 + B);
  }

  push(block: ArrayLike<number>): void {
    const h = this.hist, B = this.B;
    h.copyWithin(0, B);
    const o = h.length - B;
    for (let i = 0; i < B; i++) h[o + i] = block[i];
  }

  /** Output block for the most recent input block with filter taps h[off .. off+L). */
  apply(h: Float32Array, off: number, out: Float64Array): void {
    const L = this.L, B = this.B, x = this.hist;
    for (let n = 0; n < B; n++) {
      let acc = 0;
      const base = n + L - 1; // tap k multiplies x[n + L - 1 - k]
      for (let k = 0; k < L; k++) acc += h[off + k] * x[base - k];
      out[n] = acc;
    }
  }

  /** Same result with time-reversed float64 taps r[j] = h[L-1-j]: y[n] = sum_j r[j] x[n+j]
   *  (contiguous forward access to both arrays, which V8 optimises better). */
  applyReversed(r: Float64Array, off: number, out: Float64Array): void {
    const L = this.L, B = this.B, x = this.hist;
    for (let n = 0; n < B; n++) {
      let a0 = 0, a1 = 0, a2 = 0, a3 = 0;
      let j = 0;
      for (; j + 3 < L; j += 4) {
        a0 += r[off + j] * x[n + j];
        a1 += r[off + j + 1] * x[n + j + 1];
        a2 += r[off + j + 2] * x[n + j + 2];
        a3 += r[off + j + 3] * x[n + j + 3];
      }
      for (; j < L; j++) a0 += r[off + j] * x[n + j];
      out[n] = (a0 + a1) + (a2 + a3);
    }
  }
}

/** Time-reversed float64 copy of a 2-ear filter (for DirectConvolver.applyReversed). */
export function reverseTaps(filter: Float32Array, L: number): Float64Array {
  const r = new Float64Array(2 * L);
  for (let ear = 0; ear < 2; ear++) for (let j = 0; j < L; j++) r[ear * L + j] = filter[ear * L + L - 1 - j];
  return r;
}

export class PartitionedConvolver {
  readonly L: number;
  readonly B: number;
  readonly P: number;
  readonly bins: number;
  private readonly fft: RealFFT;
  private readonly prev: Float64Array;
  private readonly frame: Float64Array;
  private readonly fdlRe: Float64Array;
  private readonly fdlIm: Float64Array;
  private head = 0;
  private readonly accRe: Float64Array;
  private readonly accIm: Float64Array;
  private readonly time: Float64Array;
  private readonly partRe: Float64Array;
  private readonly partIm: Float64Array;
  private readonly seg: Float64Array;

  constructor(L: number, B: number) {
    if (L % B !== 0) throw new Error("filter length must be a multiple of the block size");
    this.L = L;
    this.B = B;
    this.P = L / B;
    this.bins = B + 1;
    this.fft = new RealFFT(2 * B);
    this.prev = new Float64Array(B);
    this.frame = new Float64Array(2 * B);
    this.fdlRe = new Float64Array(this.P * this.bins);
    this.fdlIm = new Float64Array(this.P * this.bins);
    this.accRe = new Float64Array(this.bins);
    this.accIm = new Float64Array(this.bins);
    this.time = new Float64Array(2 * B);
    this.partRe = new Float64Array(this.bins);
    this.partIm = new Float64Array(this.bins);
    this.seg = new Float64Array(B);
  }

  /** Partition spectra of a 2-ear filter (left then right, each L taps). */
  partition(filter: Float32Array): FilterSpectra {
    const { P, B, L, bins } = this;
    const re = new Float64Array(2 * P * bins), im = new Float64Array(2 * P * bins);
    for (let ear = 0; ear < 2; ear++) {
      for (let p = 0; p < P; p++) {
        for (let i = 0; i < B; i++) this.seg[i] = filter[ear * L + p * B + i];
        this.fft.forward(this.seg, this.partRe, this.partIm, B);
        const o = (ear * P + p) * bins;
        re.set(this.partRe, o);
        im.set(this.partIm, o);
      }
    }
    return { re, im };
  }

  push(block: ArrayLike<number>): void {
    const B = this.B, f = this.frame;
    for (let i = 0; i < B; i++) { f[i] = this.prev[i]; f[B + i] = block[i]; this.prev[i] = block[i]; }
    this.head = (this.head + 1) % this.P;
    this.fft.forward(f, this.partRe, this.partIm, 2 * B);
    const o = this.head * this.bins;
    this.fdlRe.set(this.partRe, o);
    this.fdlIm.set(this.partIm, o);
  }

  apply(H: FilterSpectra, ear: number, out: Float64Array): void {
    const { P, bins, B } = this;
    const ar = this.accRe, ai = this.accIm;
    ar.fill(0); ai.fill(0);
    for (let p = 0; p < P; p++) {
      const slot = ((this.head - p) % P + P) % P;
      const xo = slot * bins, ho = (ear * P + p) * bins;
      const xr = this.fdlRe, xi = this.fdlIm, hr = H.re, hi = H.im;
      for (let k = 0; k < bins; k++) {
        const a = xr[xo + k], b = xi[xo + k], c = hr[ho + k], d = hi[ho + k];
        ar[k] += a * c - b * d;
        ai[k] += a * d + b * c;
      }
    }
    this.fft.inverse(ar, ai, this.time);
    for (let i = 0; i < B; i++) out[i] = this.time[B + i];
  }
}
