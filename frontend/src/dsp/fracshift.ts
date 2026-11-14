import { RealFFT } from "./fft";

/**
 * Band-limited circular fractional delay on an N-point grid – mirrors
 * hrir_preprocessing.frac_shift(): bins k < N/2 are multiplied by e^{-j 2π k s / N},
 * the (real) Nyquist bin by cos(π s).
 */
export class FracShifter {
  readonly n: number;
  private readonly fft: RealFFT;
  private readonly re: Float64Array;
  private readonly im: Float64Array;
  private readonly tmp: Float64Array;

  constructor(n: number) {
    this.n = n;
    this.fft = new RealFFT(n);
    this.re = new Float64Array(n / 2 + 1);
    this.im = new Float64Array(n / 2 + 1);
    this.tmp = new Float64Array(n);
  }

  /** Delay `x` (zero-padded to N) by `shift` samples; write the first out.length samples. */
  shift(x: ArrayLike<number>, shift: number, out: Float32Array | Float64Array, xLength = x.length): void {
    const n = this.n, half = n / 2;
    this.fft.forward(x, this.re, this.im, xLength);
    const re = this.re, im = this.im;
    const w = (-2 * Math.PI * shift) / n;
    // Phasor recurrence e^{jwk} = e^{jw(k-1)} e^{jw}; re-seeded every 64 bins to bound rounding drift.
    const c1 = Math.cos(w), s1 = Math.sin(w);
    let c = 1, s = 0;
    for (let k = 0; k < half; k++) {
      if ((k & 63) === 0) { c = Math.cos(w * k); s = Math.sin(w * k); }
      const r = re[k], i = im[k];
      re[k] = r * c - i * s;
      im[k] = r * s + i * c;
      const cn = c * c1 - s * s1;
      s = c * s1 + s * c1;
      c = cn;
    }
    re[half] *= Math.cos(Math.PI * shift);
    im[half] = 0;
    this.fft.inverse(re, im, this.tmp);
    const m = Math.min(out.length, n);
    for (let i = 0; i < m; i++) out[i] = this.tmp[i];
  }
}
