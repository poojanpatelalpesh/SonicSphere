/** Instant-attack, exponential-release peak limiter (|out| ≤ threshold). Mirrors PeakLimiter in Python. */
export class PeakLimiter {
  threshold: number;
  private coef: number;
  g = 1;
  minGain = 1;

  constructor(fs: number, threshold = 0.98, releaseMs = 80) {
    this.threshold = threshold;
    this.coef = Math.exp(-1 / (releaseMs * 1e-3 * fs));
  }

  /** In-place on two channels of equal length. Returns the smallest gain applied in this call. */
  process(l: Float64Array | Float32Array, r: Float64Array | Float32Array, n = l.length): number {
    let g = this.g;
    const thr = this.threshold, c = this.coef;
    let minG = g;
    for (let i = 0; i < n; i++) {
      const pk = Math.max(Math.abs(l[i]), Math.abs(r[i]));
      const target = pk > thr ? thr / pk : 1;
      g = target < g ? target : target + (g - target) * c;
      l[i] *= g;
      r[i] *= g;
      if (g < minG) minG = g;
    }
    this.g = g;
    if (minG < this.minGain) this.minGain = minG;
    return minG;
  }
}
