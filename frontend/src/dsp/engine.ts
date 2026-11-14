/**
 * Block-based binaural engine – the algorithm that runs inside the AudioWorklet and the
 * exact mirror of backend/sonicsphere/audio_renderer.BlockEngine.
 *
 * For every block of B samples and every virtual source (voice):
 *   1. push the input block into the convolver history,
 *   2. quantise the source direction; if the cache key changed, look up / interpolate the
 *      new HRIR pair,
 *   3. y = conv(x, h_new); if the filter changed, also y_old = conv(x, h_old) and crossfade
 *      linearly over the block: y = (1−r)·y_old + r·y_new, r[n] = (n+1)/B,
 *   4. apply a linear gain ramp from the previous to the new gain,
 * then sum the voices and run the peak limiter.
 */

import { RAD, rotateAboutUp, worldToHead, type Vec3 } from "./coordinates";
import { DirectConvolver, PartitionedConvolver, reverseTaps, type ConvMethod, type FilterSpectra } from "./convolver";
import type { FilterResult, HRIRInterpolator } from "./interpolation";
import { PeakLimiter } from "./limiter";
import type { Trajectory } from "./trajectory";

export const MIN_DISTANCE_M = 0.25;

export function distanceGain(r: number, enabled = true, refM = 1): number {
  if (!enabled) return 1;
  return Math.min(2, Math.max(0.25, refM / Math.max(r, MIN_DISTANCE_M)));
}

interface Voice {
  direct: DirectConvolver | null;
  part: PartitionedConvolver | null;
  hasKey: boolean;      // false until the first filter, true afterwards
  stale: boolean;       // force a (crossfaded) refresh on the next block
  kAz: number;          // current grid key (degrees)
  kEl: number;
  cur: FilterResult | null;
  gain: number | null;
  hist: Float64Array; // last L samples of input (to rebuild a convolver after a method switch)
}

export interface EngineStats {
  blocks: number;
  filterUpdates: number;
  crossfades: number;
}

export interface EngineOptions {
  blockSize: number;
  method: ConvMethod;
  crossfade?: boolean;
  limiterThreshold?: number;
  limiterReleaseMs?: number;
  voices?: number;
  recordKeys?: boolean;
}

export class BinauralEngine {
  readonly interp: HRIRInterpolator;
  readonly B: number;
  readonly L: number;
  method: ConvMethod;
  crossfade: boolean;
  readonly limiter: PeakLimiter;
  private voices: Voice[] = [];
  readonly ramp: Float64Array;
  readonly stats: EngineStats = { blocks: 0, filterUpdates: 0, crossfades: 0 };
  readonly keys: [number, number][] = [];
  recordKeys: boolean;
  private readonly yNew: [Float64Array, Float64Array];
  private readonly yOld: [Float64Array, Float64Array];
  private readonly outL: Float64Array;
  private readonly outR: Float64Array;
  private readonly spectraOwner: PartitionedConvolver;

  constructor(interp: HRIRInterpolator, opts: EngineOptions) {
    this.interp = interp;
    this.B = opts.blockSize;
    this.L = interp.pack.L;
    this.method = opts.method;
    this.crossfade = opts.crossfade ?? true;
    this.limiter = new PeakLimiter(interp.pack.header.fs, opts.limiterThreshold ?? interp.pack.header.limiterThreshold,
                                   opts.limiterReleaseMs ?? interp.pack.header.limiterReleaseMs);
    this.recordKeys = opts.recordKeys ?? false;
    this.ramp = new Float64Array(this.B);
    for (let n = 0; n < this.B; n++) this.ramp[n] = (n + 1) / this.B;
    this.yNew = [new Float64Array(this.B), new Float64Array(this.B)];
    this.yOld = [new Float64Array(this.B), new Float64Array(this.B)];
    this.outL = new Float64Array(this.B);
    this.outR = new Float64Array(this.B);
    this.spectraOwner = new PartitionedConvolver(this.L, this.B);
    this.setVoiceCount(opts.voices ?? 1);
  }

  get voiceCount(): number { return this.voices.length; }

  currentFilter(v = 0): FilterResult | null { return this.voices[v]?.cur ?? null; }

  setVoiceCount(n: number): void {
    while (this.voices.length < n) {
      this.voices.push({ direct: null, part: null, hasKey: false, stale: false, kAz: NaN, kEl: NaN, cur: null, gain: null, hist: new Float64Array(this.L) });
      this.makeConvolver(this.voices[this.voices.length - 1]);
    }
    this.voices.length = n;
  }

  private makeConvolver(v: Voice): void {
    v.direct = this.method === "direct" ? new DirectConvolver(this.L, this.B) : null;
    v.part = this.method === "partitioned" ? new PartitionedConvolver(this.L, this.B) : null;
    // Replay the input history so the new convolver continues seamlessly.
    for (let o = 0; o < this.L; o += this.B) {
      const blk = v.hist.subarray(o, o + this.B);
      v.direct?.push(blk);
      v.part?.push(blk);
    }
  }

  setMethod(m: ConvMethod): void {
    if (m === this.method) return;
    this.method = m;
    for (const v of this.voices) this.makeConvolver(v);
  }

  /** Force a filter refresh (crossfaded) on the next block, e.g. after a dataset change. */
  invalidate(): void {
    for (const v of this.voices) if (v.hasKey) v.stale = true;
  }

  private spectra(f: FilterResult): FilterSpectra {
    if (!f.spectra) f.spectra = this.spectraOwner.partition(f.filter);
    return f.spectra as FilterSpectra;
  }

  private applyFilter(v: Voice, f: FilterResult, out: [Float64Array, Float64Array]): void {
    if (v.direct) {
      const r = (f.reversed ??= reverseTaps(f.filter, this.L));
      v.direct.applyReversed(r, 0, out[0]);
      v.direct.applyReversed(r, this.L, out[1]);
    } else {
      const s = this.spectra(f);
      v.part!.apply(s, 0, out[0]);
      v.part!.apply(s, 1, out[1]);
    }
  }

  /**
   * Process one block. inputs[i]: voice i's B samples; dirs[i]: its head-frame direction;
   * gains[i]: its target gain. Output written to outL/outR (length ≥ B).
   */
  processBlock(inputs: ArrayLike<number>[], dirs: ArrayLike<number>[], gains: number[],
               outL: Float32Array | Float64Array, outR: Float32Array | Float64Array): void {
    const B = this.B, L = this.L;
    const oL = this.outL, oR = this.outR;
    oL.fill(0); oR.fill(0);
    for (let vi = 0; vi < this.voices.length; vi++) {
      const v = this.voices[vi];
      const xb = inputs[vi];
      v.hist.copyWithin(0, B);
      for (let i = 0; i < B; i++) v.hist[L - B + i] = xb[i];
      v.direct?.push(xb);
      v.part?.push(xb);
      // Grid key without allocating (same arithmetic as cartToSph + quantizeDirection).
      const d = dirs[vi];
      let az = (Math.atan2(d[1], d[0]) * RAD) % 360;
      if (az < 0) az += 360;
      if (az >= 360) az -= 360;
      const el = Math.atan2(d[2], Math.hypot(d[0], d[1])) * RAD;
      const q = this.interp.quantum;
      let elQ = Math.min(90, Math.max(-90, Math.floor(el / q + 0.5) * q));
      let azQ = Math.floor(az / q + 0.5) * q;
      if (azQ >= 360 - 1e-9) azQ -= 360;
      if (Math.abs(elQ) >= 90 - 1e-9) azQ = 0;
      if (elQ === 0) elQ = 0; // normalise -0
      if (this.recordKeys) this.keys.push([azQ, elQ]);
      const yN = this.yNew;
      if (!v.hasKey || v.stale || azQ !== v.kAz || elQ !== v.kEl) {
        const f = this.interp.query(dirs[vi]);
        this.applyFilter(v, f, yN);
        if (v.hasKey && v.cur && this.crossfade) {
          this.applyFilter(v, v.cur, this.yOld);
          const r = this.ramp, yO = this.yOld;
          for (let n = 0; n < B; n++) {
            yN[0][n] = yO[0][n] * (1 - r[n]) + yN[0][n] * r[n];
            yN[1][n] = yO[1][n] * (1 - r[n]) + yN[1][n] * r[n];
          }
          this.stats.crossfades++;
        }
        v.hasKey = true; v.stale = false; v.kAz = azQ; v.kEl = elQ;
        v.cur = f;
        this.stats.filterUpdates++;
      } else {
        this.applyFilter(v, v.cur!, yN);
      }
      const g = gains[vi];
      const g0 = v.gain === null ? g : v.gain;
      const dg = g - g0, r = this.ramp;
      for (let n = 0; n < B; n++) {
        const gn = g0 + dg * r[n];
        oL[n] += yN[0][n] * gn;
        oR[n] += yN[1][n] * gn;
      }
      v.gain = g;
    }
    this.stats.blocks++;
    this.limiter.process(oL, oR, B);
    for (let n = 0; n < B; n++) { outL[n] = oL[n]; outR[n] = oR[n]; }
  }
}

export interface Pose { position: Vec3; yaw: number; pitch: number }

/** Directions and gains for every voice of a source at world position `src`. */
export function voiceParams(src: ArrayLike<number>, pose: Pose, offsets: number[], gain: number,
                            distanceAttenuation: boolean): { dirs: Vec3[]; gains: number[]; distance: number; head: Vec3 } {
  const d = worldToHead(src, pose.position, pose.yaw, pose.pitch);
  const r = Math.hypot(d[0], d[1], d[2]);
  const head: Vec3 = r > 1e-9 ? [d[0] / r, d[1] / r, d[2] / r] : [1, 0, 0];
  const g = gain * distanceGain(r, distanceAttenuation);
  const dirs = offsets.map((o) => (o ? rotateAboutUp(head, o) : head));
  return { dirs, gains: offsets.map(() => g), distance: r, head };
}

/** Offline render of a moving source with the real-time algorithm (tests / benchmarks). */
export function renderTrajectory(engine: BinauralEngine, signals: Float64Array[] | Float32Array[], offsets: number[],
                                 traj: Trajectory, pose: Pose, gain: number, distanceAttenuation = true): [Float64Array, Float64Array] {
  const B = engine.B, fs = engine.interp.pack.header.fs;
  const n = Math.max(...signals.map((s) => s.length));
  const nb = Math.ceil(n / B);
  const L = new Float64Array(nb * B), R = new Float64Array(nb * B);
  const blocks = signals.map(() => new Float64Array(B));
  const tmpL = new Float64Array(B), tmpR = new Float64Array(B);
  const pos: Vec3 = [0, 0, 0];
  for (let b = 0; b < nb; b++) {
    const frame = b * B;
    signals.forEach((s, i) => {
      for (let k = 0; k < B; k++) blocks[i][k] = frame + k < s.length ? s[frame + k] : 0;
    });
    traj.positionAt(frame / fs, pos);
    const { dirs, gains } = voiceParams(pos, pose, offsets, gain, distanceAttenuation);
    engine.processBlock(blocks, dirs, gains, tmpL, tmpR);
    L.set(tmpL, frame);
    R.set(tmpR, frame);
  }
  return [L.subarray(0, n), R.subarray(0, n)];
}
