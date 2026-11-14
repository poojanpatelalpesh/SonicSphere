/**
 * SonicSphere AudioWorklet processor: the real-time binaural renderer.
 *
 * Per render quantum (128 frames):
 *   source player (looping buffer, sample-accurate transport with 256-sample fades)
 *   → source position (manual position or trajectory evaluated on the audio clock)
 *   → world → head transform → per-voice direction
 *   → BinauralEngine (direction lookup, interpolation, crossfade, convolution, limiter)
 *   → output 0: binaural stereo (or, with 3D switched off, the plain sound at matched loudness)
 *   → output 1: dry mono send (feeds the room-ambience convolver on the main graph)
 *
 * The processor allocates nothing per block (garbage collection on the audio thread
 * causes dropouts on slower machines), measures its own load and coarsens the filter
 * grid automatically if the machine cannot keep up.
 */
/// <reference path="./worklet-env.d.ts" />

import type { Vec3 } from "../dsp/coordinates";
import { listenerBasis } from "../dsp/coordinates";
import type { ConvMethod } from "../dsp/convolver";
import { BinauralEngine, distanceGain } from "../dsp/engine";
import { parsePack, type FilterPack } from "../dsp/filterPack";
import { HRIRInterpolator, type InterpMethod } from "../dsp/interpolation";
import { cartToSph } from "../dsp/coordinates";
import { Trajectory } from "../dsp/trajectory";
import type { FromWorklet, ToWorklet, VoiceStatus, WorkletStatus } from "./protocol";

const FADE = 256;
const STATUS_EVERY = 8;      // blocks (~23 ms at 44.1 kHz)
const ACK_EVERY = 4;
const MAX_VOICES = 2;
const ECONOMY_ON = 0.55;     // load (fraction of the block deadline) that triggers economy mode
const ECONOMY_OFF = 0.2;
const ECONOMY_QUANTUM = 1.0; // degrees

const perf: { now(): number } | null = (() => {
  try {
    const p = (globalThis as unknown as { performance?: { now(): number } }).performance;
    return p && typeof p.now === "function" ? p : null;
  } catch {
    return null;
  }
})();
const clockMs = perf ? () => perf.now() : () => Date.now();

class SpatializerProcessor extends AudioWorkletProcessor {
  private pack: FilterPack | null = null;
  private interp: HRIRInterpolator | null = null;
  private engine: BinauralEngine | null = null;
  private method: ConvMethod = "partitioned";
  private interpMethod: InterpMethod = "barycentric_aligned";
  private crossfade = true;
  private distanceAttenuation = true;
  private volume = 1;
  private spatial = true;
  private mix = 1;               // 1 = binaural, 0 = plain (ramped)
  private plainGain = 0.5;       // loudness-matched gain for the plain (3D off) signal

  // source
  private signals: Float32Array[] = [];
  private offsets: number[] = [0];
  private sourceId: string | null = null;
  private sourceLoop = true;
  private playhead = 0;
  private playing = false;
  private fade = 0;
  private fadeTarget = 0;
  private stopAfterFade = false;
  private pendingRewind = false;

  // geometry
  private manualPos: Vec3 = [-1.13, 1.15, -1.13];
  private listenerPos: Vec3 = [0, 1.15, 0];
  private basis = listenerBasis(0, 0);
  private traj: Trajectory | null = null;
  private trajPlaying = false;
  private trajTime = 0;
  private curPos: Vec3 = [0, 0, 0];

  // preallocated per-block buffers
  private readonly inBlocks: Float32Array[] = Array.from({ length: MAX_VOICES }, () => new Float32Array(128));
  private activeBlocks: Float32Array[] = [this.inBlocks[0]];
  private readonly dirs: Vec3[] = Array.from({ length: MAX_VOICES }, () => [1, 0, 0] as Vec3);
  private activeDirs: Vec3[] = [this.dirs[0]];
  private gains: number[] = [1];
  private readonly head: Vec3 = [1, 0, 0];

  // stats
  private posCount = 0;
  private blockCounter = 0;
  private tSum = 0;
  private tMax = 0;
  private tN = 0;
  private loadEma = 0;
  private economy = false;
  private overruns = 0;
  private limMin = 1;
  private pkL = 0;
  private pkR = 0;
  private pkIn = 0;
  private lastDistance = 1;

  constructor() {
    super();
    this.port.onmessage = (e: MessageEvent<ToWorklet>) => {
      try {
        this.onMessage(e.data);
      } catch (err) {
        this.post({ type: "error", message: String((err as Error)?.message ?? err) });
      }
    };
    this.post({ type: "ready", sampleRate, blockSize: 128, timer: perf !== null });
  }

  private post(m: FromWorklet, transfer: Transferable[] = []) {
    this.port.postMessage(m, transfer);
  }

  private setVoices(n: number) {
    n = Math.max(1, Math.min(MAX_VOICES, n));
    this.activeBlocks = this.inBlocks.slice(0, n);
    this.activeDirs = this.dirs.slice(0, n);
    this.gains = new Array(n).fill(1);
    this.engine?.setVoiceCount(n);
  }

  /** Loudness-matched gain for 3D-off: RMS ear-filter gain averaged over all directions. */
  private computePlainGain(pack: FilterPack) {
    const { measured, M, L } = pack;
    let e = 0;
    for (let i = 0; i < measured.length; i++) e += measured[i] * measured[i];
    this.plainGain = Math.sqrt(e / (M * 2)) * pack.header.gain;
    void L;
  }

  private onMessage(m: ToWorklet) {
    switch (m.type) {
      case "pack": {
        const pack = parsePack(m.pack);
        if (pack.header.blockSize !== 128) throw new Error(`Filter pack block size ${pack.header.blockSize} ≠ render quantum 128`);
        if (Math.abs(pack.header.fs - sampleRate) > 0.5) {
          throw new Error(`Filter pack is for ${pack.header.fs} Hz but the AudioContext runs at ${sampleRate} Hz`);
        }
        this.pack = pack;
        this.method = m.method;
        this.interpMethod = m.interp;
        this.computePlainGain(pack);
        const old = this.engine;
        this.interp = new HRIRInterpolator(pack, this.interpMethod);
        if (this.economy) this.interp.setQuantum(Math.max(ECONOMY_QUANTUM, pack.header.directionQuantumDeg));
        this.engine = new BinauralEngine(this.interp, { blockSize: 128, method: this.method, crossfade: this.crossfade,
                                                         voices: this.activeBlocks.length });
        if (old) this.engine.limiter.g = old.limiter.g;
        break;
      }
      case "source": {
        this.signals = m.signals.slice(0, MAX_VOICES);
        this.offsets = m.offsets.slice(0, MAX_VOICES);
        this.sourceId = m.id;
        this.sourceLoop = m.loop;
        this.playhead = 0;
        this.setVoices(this.signals.length);
        if (this.playing) { this.fade = 0; this.fadeTarget = 1; }
        break;
      }
      case "transport":
        if (m.action === "play") {
          if (this.signals.length === 0) break;
          if (!this.playing || this.stopAfterFade) { this.playing = true; this.stopAfterFade = false; this.fadeTarget = 1; }
        } else {
          this.fadeTarget = 0;
          this.stopAfterFade = true;
          if (m.action === "stop") this.pendingRewind = true;
        }
        break;
      case "position":
        this.manualPos = m.pos;
        if (++this.posCount % ACK_EVERY === 0) this.post({ type: "ack", seq: m.seq, t: m.t, frame: currentFrame });
        break;
      case "listener":
        this.listenerPos = m.position;
        this.basis = listenerBasis(m.yaw, m.pitch);
        break;
      case "trajectory":
        if (m.samples) {
          const keep = this.traj ? this.trajTime : 0;
          this.traj = new Trajectory(m.samples, m.duration, m.loop);
          this.trajTime = Math.min(keep, m.loop ? Infinity : m.duration);
        } else {
          this.traj = null;
          this.trajPlaying = false;
        }
        break;
      case "trajectoryTransport":
        if (!this.traj) break;
        if (m.action === "play") this.trajPlaying = true;
        else if (m.action === "pause") this.trajPlaying = false;
        else if (m.action === "stop") { this.trajPlaying = false; this.trajTime = 0; }
        else if (m.action === "restart") { this.trajTime = 0; this.trajPlaying = true; }
        break;
      case "trajectoryDuration":
        if (this.traj && m.duration > 0) {
          const ph = this.traj.phaseAt(this.trajTime);
          this.traj.durationS = m.duration;
          this.trajTime = ph * m.duration;
        }
        break;
      case "settings":
        if (m.volume !== undefined) this.volume = m.volume;
        if (m.spatial !== undefined) this.spatial = m.spatial;
        if (m.crossfade !== undefined) { this.crossfade = m.crossfade; if (this.engine) this.engine.crossfade = m.crossfade; }
        if (m.distanceAttenuation !== undefined) this.distanceAttenuation = m.distanceAttenuation;
        if (m.loop !== undefined) this.sourceLoop = m.loop;
        if (m.method !== undefined) { this.method = m.method; this.engine?.setMethod(m.method); }
        if (m.interp !== undefined && m.interp !== this.interpMethod) {
          this.interpMethod = m.interp;
          this.interp?.setMethod(m.interp);
          this.engine?.invalidate();
        }
        break;
      case "snapshot": {
        const f = this.engine?.currentFilter(0);
        if (f) {
          const copy = f.filter.slice();
          this.post({ type: "snapshot", filter: copy, toa: f.toa, L: this.pack!.L, status: this.voiceStatus(0) }, [copy.buffer]);
        }
        break;
      }
      case "ping":
        this.post({ type: "pong", seq: m.seq, t: m.t, frame: currentFrame });
        break;
    }
  }

  private voiceStatus(i: number): VoiceStatus | null {
    const f = this.engine?.currentFilter(i);
    if (!f) return null;
    const rs = cartToSph(f.w.rendered);
    return {
      status: f.w.status, az: f.az, el: f.el, renderedAz: rs.az, renderedEl: rs.el,
      indices: f.w.indices.slice(0, f.w.n), weights: f.w.weights.slice(0, f.w.n), toa: f.toa,
      nearestIndex: f.w.nearestIndex, nearestAngleDeg: f.w.nearestAngleDeg, maxEdgeDeg: f.w.maxEdgeDeg,
      lateralDeg: f.w.lateralDeg, polarLoDeg: f.w.polarLoDeg, polarHiDeg: f.w.polarHiDeg,
    };
  }

  private readSource(n: number) {
    const sigs = this.signals;
    const len = sigs.length ? sigs[0].length : 0;
    const nv = this.activeBlocks.length;
    for (let i = 0; i < n; i++) {
      if (this.fade < this.fadeTarget) this.fade = Math.min(this.fadeTarget, this.fade + 1 / FADE);
      else if (this.fade > this.fadeTarget) this.fade = Math.max(this.fadeTarget, this.fade - 1 / FADE);
      const active = this.playing && len > 0;
      for (let v = 0; v < nv; v++) this.activeBlocks[v][i] = active && v < sigs.length ? sigs[v][this.playhead] * this.fade : 0;
      if (active) {
        this.playhead++;
        if (this.playhead >= len) {
          if (this.sourceLoop) this.playhead = 0;
          else { this.playhead = 0; this.playing = false; this.fade = 0; this.fadeTarget = 0; this.post({ type: "ended" }); }
        }
      }
      if (this.stopAfterFade && this.fade === 0) {
        this.playing = false;
        this.stopAfterFade = false;
        if (this.pendingRewind) { this.playhead = 0; this.pendingRewind = false; }
      }
    }
  }

  /** Head-frame directions and gains for all voices, written into preallocated arrays. */
  private updateGeometry() {
    const p = this.curPos, l = this.listenerPos, [f, le, u] = this.basis;
    const d0 = p[0] - l[0], d1 = p[1] - l[1], d2 = p[2] - l[2];
    const x = d0 * f[0] + d1 * f[1] + d2 * f[2];
    const y = d0 * le[0] + d1 * le[1] + d2 * le[2];
    const z = d0 * u[0] + d1 * u[1] + d2 * u[2];
    const r = Math.hypot(x, y, z);
    const h = this.head;
    if (r > 1e-9) { h[0] = x / r; h[1] = y / r; h[2] = z / r; } else { h[0] = 1; h[1] = 0; h[2] = 0; }
    this.lastDistance = r;
    const g = this.pack!.header.gain * this.volume * distanceGain(r, this.distanceAttenuation);
    for (let v = 0; v < this.activeDirs.length; v++) {
      const dv = this.activeDirs[v];
      const off = this.offsets[v] ?? 0;
      if (off) {
        const a = (off * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a);
        dv[0] = c * h[0] - s * h[1]; dv[1] = s * h[0] + c * h[1]; dv[2] = h[2];
      } else { dv[0] = h[0]; dv[1] = h[1]; dv[2] = h[2]; }
      this.gains[v] = g;
    }
  }

  process(_inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    const out = outputs[0];
    const L = out[0], R = out[1] ?? out[0];
    const send = outputs[1]?.[0];
    const n = L.length;
    if (!this.engine || n !== 128) {
      L.fill(0); R.fill(0); send?.fill(0);
      return true;
    }
    const t0 = clockMs();
    this.readSource(n);

    if (this.traj && (this.trajPlaying || this.trajTime > 0)) {
      this.traj.positionAt(this.trajTime, this.curPos);
      if (this.trajPlaying) {
        this.trajTime += n / sampleRate;
        if (!this.traj.loop && this.trajTime >= this.traj.durationS) {
          this.trajTime = this.traj.durationS;
          this.trajPlaying = false;
          this.post({ type: "trajectoryEnded" });
        }
      }
    } else {
      this.curPos[0] = this.manualPos[0]; this.curPos[1] = this.manualPos[1]; this.curPos[2] = this.manualPos[2];
    }
    this.updateGeometry();
    this.engine.processBlock(this.activeBlocks, this.activeDirs, this.gains, L, R);

    // 3D on/off: crossfade (over ~6 ms) to the plain sound, equal in both ears, at matched loudness.
    const target = this.spatial ? 1 : 0;
    const nv = this.activeBlocks.length;
    const plain = this.plainGain * this.volume * distanceGain(this.lastDistance, this.distanceAttenuation) / nv;
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let v = 0; v < nv; v++) s += this.activeBlocks[v][i];
      if (this.mix !== target) this.mix = target > this.mix ? Math.min(1, this.mix + 1 / FADE) : Math.max(0, this.mix - 1 / FADE);
      if (this.mix < 1) {
        const p = s * plain;
        L[i] = this.mix * L[i] + (1 - this.mix) * p;
        R[i] = this.mix * R[i] + (1 - this.mix) * p;
      }
      if (send) send[i] = (s / nv) * this.volume;
      const a = Math.abs(L[i]), b = Math.abs(R[i]), c = Math.abs(s);
      if (a > this.pkL) this.pkL = a;
      if (b > this.pkR) this.pkR = b;
      if (c > this.pkIn) this.pkIn = c;
    }
    if (this.engine.limiter.g < this.limMin) this.limMin = this.engine.limiter.g;

    // Load measurement (Date.now has 1 ms resolution; its mean over many blocks is unbiased).
    const dt = (clockMs() - t0) * 1000;
    const deadline = (n / sampleRate) * 1e6;
    this.tSum += dt; this.tN++;
    if (dt > this.tMax) this.tMax = dt;
    if (perf && dt > deadline) this.overruns++;
    this.loadEma += 0.002 * (dt / deadline - this.loadEma);   // ~1.5 s time constant
    if (!this.economy && this.loadEma > ECONOMY_ON && this.blockCounter > 700) {
      this.economy = true;
      this.interp?.setQuantum(Math.max(ECONOMY_QUANTUM, this.pack!.header.directionQuantumDeg));
    } else if (this.economy && this.loadEma < ECONOMY_OFF) {
      this.economy = false;
      this.interp?.setQuantum(this.pack!.header.directionQuantumDeg);
    }
    if (++this.blockCounter % STATUS_EVERY === 0) this.postStatus();
    return true;
  }

  private postStatus() {
    const eng = this.engine!, it = this.interp!;
    const voices: VoiceStatus[] = [];
    for (let i = 0; i < eng.voiceCount; i++) { const s = this.voiceStatus(i); if (s) voices.push(s); }
    const msg: WorkletStatus = {
      type: "status", frame: currentFrame, time: currentTime, sampleRate, blockSize: 128, filterLength: this.pack!.L,
      transport: { playing: this.playing && !this.stopAfterFade, position: this.signals.length ? this.playhead / sampleRate : 0,
                   duration: this.signals.length ? this.signals[0].length / sampleRate : 0, sourceId: this.sourceId,
                   loop: this.sourceLoop },
      trajectory: { active: this.traj !== null && (this.trajPlaying || this.trajTime > 0), playing: this.trajPlaying,
                    phase: this.traj ? this.traj.phaseAt(this.trajTime) : 0, duration: this.traj?.durationS ?? 0,
                    loop: this.traj?.loop ?? true },
      position: [this.curPos[0], this.curPos[1], this.curPos[2]],
      distance: this.lastDistance,
      headDir: [this.head[0], this.head[1], this.head[2]],
      voices,
      method: eng.method,
      interp: it.method,
      spatial: this.spatial,
      stats: { ...eng.stats, cacheHits: it.cacheHits, cacheMisses: it.cacheMisses, cacheSize: it.cacheSize },
      timing: { available: true, estimated: perf === null, meanUs: this.tN ? this.tSum / this.tN : 0, maxUs: this.tMax,
                overruns: this.overruns, windowBlocks: this.tN, load: this.loadEma, economy: this.economy, quantum: it.quantum },
      limiterMinGain: this.limMin,
      peaks: { left: this.pkL, right: this.pkR, input: this.pkIn },
    };
    this.post(msg);
    this.tSum = 0; this.tN = 0; this.tMax = 0; this.limMin = 1; this.pkL = 0; this.pkR = 0; this.pkIn = 0;
  }
}

registerProcessor("sonicsphere-spatializer", SpatializerProcessor);
