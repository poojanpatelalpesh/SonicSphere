/**
 * Main-thread controller of the real-time renderer.
 *
 * Owns the AudioContext and the AudioWorkletNode, forwards control messages (source
 * position, listener pose, trajectory, transport, settings) and collects live status.
 * No DSP happens here: the worklet does all audio processing.
 */

import workletUrl from "./spatializer.worklet.ts?worker&url";
import type { Vec3 } from "../dsp/coordinates";
import type { ConvMethod } from "../dsp/convolver";
import type { InterpMethod } from "../dsp/interpolation";
import type { FromWorklet, ToWorklet, VoiceStatus, WorkletStatus } from "./protocol";

export type AudioState = "off" | "starting" | "running" | "suspended" | "interrupted" | "closed" | "error";

export interface EngineListener {
  onStatus?(s: WorkletStatus): void;
  onState?(s: AudioState, detail?: string): void;
  onLatency?(rttMs: number): void;
  onEnded?(): void;
  onTrajectoryEnded?(): void;
  onError?(msg: string): void;
}

interface SourceMsg { id: string; signals: Float32Array[]; offsets: number[]; loop: boolean }

/** Ambience level relative to the dry send (derived from the loudness-matched direct gain ≈ 0.53 for
 *  CIPIC: −10 dB re the direct sound at 1.5 m ⇒ 0.316 × 0.667 × 0.53 ≈ 0.11). */
const AMBIENCE_LEVEL = 0.11;

/** Small-room response: sparse early reflections (5–45 ms) and an exponentially decaying,
 *  progressively darker diffuse tail (RT60 ≈ 0.55 s), independent in each ear, unit energy. */
function makeRoomResponse(ctx: BaseAudioContext): AudioBuffer {
  const sr = ctx.sampleRate, n = Math.round(sr * 0.75);
  const buf = ctx.createBuffer(2, n, sr);
  let seed = 12345;
  const rnd = () => { seed = (1103515245 * seed + 12345) >>> 0; return seed / 2 ** 32 * 2 - 1; };
  const tau = 0.55 / 6.91;
  for (let c = 0; c < 2; c++) {
    const h = buf.getChannelData(c);
    for (let k = 0; k < 10; k++) {
      const t = 0.005 + (0.04 * (k + 0.5 + 0.4 * rnd())) / 10;
      h[Math.round(t * sr)] += (0.6 - 0.04 * k) * (rnd() > 0 ? 1 : -1);
    }
    let lp = 0;
    const start = Math.round(0.012 * sr);
    for (let i = start; i < n; i++) {
      const t = i / sr;
      const a = 0.6 * Math.exp(-t / 0.18) + 0.08;               // tail gets darker over time
      lp += a * (rnd() - lp);
      h[i] += 0.35 * lp * Math.exp(-t / tau);
    }
    let e = 0;
    for (let i = 0; i < n; i++) e += h[i] * h[i];
    const g = 1 / Math.sqrt(e);
    for (let i = 0; i < n; i++) h[i] *= g;
  }
  return buf;
}

export class AudioEngine {
  ctx: AudioContext | null = null;
  node: AudioWorkletNode | null = null;
  analyserL: AnalyserNode | null = null;
  analyserR: AnalyserNode | null = null;
  timerInWorklet = false;
  private bus: GainNode | null = null;         // binaural + ambience
  private master: GainNode | null = null;      // → destination (analysers tap here: what you actually hear)
  private ambienceGain: GainNode | null = null;
  private ambienceOn = true;
  private listener: EngineListener;
  private seq = 0;
  private rtts: number[] = [];
  private pingTimer: number | null = null;
  private snapshotWaiters: ((m: Extract<FromWorklet, { type: "snapshot" }>) => void)[] = [];
  // Last state sent, replayed if the processor must be rebuilt.
  private lastPack: ArrayBuffer | null = null;
  private lastSource: SourceMsg | null = null;
  private lastMsgs = new Map<string, ToWorklet>();
  private packParams: { method: ConvMethod; interp: InterpMethod } = { method: "partitioned", interp: "barycentric_aligned" };
  private previewNode: AudioBufferSourceNode | null = null;
  private ambienceConv: ConvolverNode | null = null;
  private checkNode: AudioBufferSourceNode | null = null;

  constructor(listener: EngineListener) {
    this.listener = listener;
  }

  get sampleRate(): number { return this.ctx?.sampleRate ?? 0; }

  /** Create the context (must be called from a user gesture). */
  async start(preferredRate: number): Promise<AudioContext> {
    if (this.ctx && this.ctx.state !== "closed") {
      await this.resume();
      return this.ctx;
    }
    this.listener.onState?.("starting");
    let ctx: AudioContext;
    try {
      ctx = new AudioContext({ sampleRate: preferredRate, latencyHint: "interactive" });
    } catch {
      ctx = new AudioContext({ latencyHint: "interactive" }); // browser refused the rate; filters get resampled
    }
    this.ctx = ctx;
    ctx.onstatechange = () => this.reportState();
    if (!ctx.audioWorklet) throw new Error("AudioWorklet is not supported by this browser");
    await ctx.audioWorklet.addModule(workletUrl);
    this.createNode();
    if (ctx.state !== "running") await ctx.resume().catch(() => undefined);
    this.reportState();
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "visible" && this.ctx?.state !== "running") void this.resume();
    });
    this.pingTimer = window.setInterval(() => this.ping(), 1000);
    return ctx;
  }

  private createNode() {
    const ctx = this.ctx!;
    const node = new AudioWorkletNode(ctx, "sonicsphere-spatializer", {
      numberOfInputs: 0, numberOfOutputs: 2, outputChannelCount: [2, 1],
    });
    node.port.onmessage = (e: MessageEvent<FromWorklet>) => this.onMessage(e.data);
    node.onprocessorerror = () => {
      this.listener.onError?.("The audio processor crashed; it has been restarted.");
      this.rebuild();
    };
    if (!this.master) {
      // Output graph (built once): bus → master → speakers, analysers on the master.
      this.bus = ctx.createGain();
      this.master = ctx.createGain();
      this.master.gain.value = 0.9;            // headroom for direct + ambience
      this.bus.connect(this.master).connect(ctx.destination);
      const split = ctx.createChannelSplitter(2);
      this.analyserL = ctx.createAnalyser();
      this.analyserR = ctx.createAnalyser();
      for (const a of [this.analyserL, this.analyserR]) { a.fftSize = 2048; a.smoothingTimeConstant = 0; }
      this.master.connect(split);
      split.connect(this.analyserL, 0);
      split.connect(this.analyserR, 1);
      // Room ambience: the dry source through a small-room response, decorrelated between the
      // ears and ~10 dB below the direct sound at 1.5 m. Reflections help sounds appear outside
      // the head; their direction-independence keeps the binaural cues of the direct sound intact.
      const conv = ctx.createConvolver();
      conv.normalize = false;
      conv.buffer = makeRoomResponse(ctx);
      this.ambienceGain = ctx.createGain();
      this.ambienceGain.gain.value = this.ambienceOn ? AMBIENCE_LEVEL : 0;
      this.ambienceConv = conv;
      conv.connect(this.ambienceGain).connect(this.bus);
    }
    node.connect(this.bus!, 0);
    node.connect(this.ambienceConv!, 1);
    this.node = node;
  }

  private rebuild() {
    try { this.node?.disconnect(); } catch { /* already gone */ }
    this.createNode();
    if (this.lastPack) this.sendPack(this.lastPack.slice(0), this.packParams.method, this.packParams.interp);
    if (this.lastSource) this.setSource(this.lastSource.id, this.lastSource.signals, this.lastSource.offsets, this.lastSource.loop);
    for (const m of this.lastMsgs.values()) this.post(m);
  }

  private reportState() {
    const s = this.ctx?.state as string | undefined;
    const map: Record<string, AudioState> = { running: "running", suspended: "suspended", interrupted: "interrupted", closed: "closed" };
    this.listener.onState?.(s ? map[s] ?? "suspended" : "off");
  }

  async resume() {
    try { await this.ctx?.resume(); } catch (e) { this.listener.onError?.(`Could not resume audio: ${(e as Error).message}`); }
    this.reportState();
  }

  private post(m: ToWorklet, transfer: Transferable[] = []) {
    this.node?.port.postMessage(m, transfer);
  }

  private remember(key: string, m: ToWorklet) {
    this.lastMsgs.set(key, m);
    this.post(m);
  }

  private onMessage(m: FromWorklet) {
    switch (m.type) {
      case "ready": this.timerInWorklet = m.timer; break;
      case "status": this.listener.onStatus?.(m); break;
      case "ack":
      case "pong": {
        const rtt = performance.now() - m.t;
        this.rtts.push(rtt);
        if (this.rtts.length > 40) this.rtts.shift();
        const sorted = [...this.rtts].sort((a, b) => a - b);
        this.listener.onLatency?.(sorted[Math.floor(sorted.length / 2)]);
        break;
      }
      case "snapshot": this.snapshotWaiters.splice(0).forEach((f) => f(m)); break;
      case "ended": this.listener.onEnded?.(); break;
      case "trajectoryEnded": this.listener.onTrajectoryEnded?.(); break;
      case "error": this.listener.onError?.(m.message); break;
    }
  }

  private ping() {
    if (this.ctx?.state === "running") this.post({ type: "ping", seq: ++this.seq, t: performance.now() });
  }

  // ------------------------------------------------------------- controls
  sendPack(pack: ArrayBuffer, method: ConvMethod, interp: InterpMethod) {
    this.lastPack = pack.slice(0);
    this.packParams = { method, interp };
    this.post({ type: "pack", pack, method, interp }, [pack]);
  }

  setSource(id: string, signals: Float32Array[], offsets: number[], loop: boolean) {
    this.lastSource = { id, signals, offsets, loop };
    const copies = signals.map((s) => s.slice());
    this.post({ type: "source", id, signals: copies, offsets, loop }, copies.map((c) => c.buffer));
  }

  transport(action: "play" | "pause" | "stop") { this.post({ type: "transport", action }); }

  setPosition(pos: Vec3) {
    const m: ToWorklet = { type: "position", pos, seq: ++this.seq, t: performance.now() };
    this.lastMsgs.set("position", m);
    this.post(m);
  }

  setListener(position: Vec3, yaw: number, pitch: number) { this.remember("listener", { type: "listener", position, yaw, pitch }); }

  setTrajectory(samples: Float64Array | null, duration: number, loop: boolean) {
    this.remember("trajectory", { type: "trajectory", samples: samples ? samples.slice() : null, duration, loop });
  }

  trajectoryTransport(action: "play" | "pause" | "stop" | "restart") { this.post({ type: "trajectoryTransport", action }); }

  setTrajectoryDuration(duration: number) { this.post({ type: "trajectoryDuration", duration }); }

  settings(s: { volume?: number; spatial?: boolean; method?: ConvMethod; interp?: InterpMethod; crossfade?: boolean; distanceAttenuation?: boolean; loop?: boolean }) {
    if (s.method) this.packParams.method = s.method;
    if (s.interp) this.packParams.interp = s.interp;
    const prev = this.lastMsgs.get("settings") as Extract<ToWorklet, { type: "settings" }> | undefined;
    this.lastMsgs.set("settings", { ...(prev ?? { type: "settings" }), ...s, type: "settings" });
    this.post({ type: "settings", ...s });
  }

  snapshot(): Promise<{ filter: Float32Array; toa: [number, number]; L: number; status: VoiceStatus | null }> {
    return new Promise((resolve) => {
      this.snapshotWaiters.push(resolve);
      this.post({ type: "snapshot" });
    });
  }

  setAmbience(on: boolean) {
    this.ambienceOn = on;
    const g = this.ambienceGain, ctx = this.ctx;
    if (g && ctx) g.gain.setTargetAtTime(on ? AMBIENCE_LEVEL : 0, ctx.currentTime, 0.05);
  }

  /** Headphone check: three short noise bursts in one ear only, straight to the output. */
  headphoneCheck(side: "left" | "right", onEnd: () => void) {
    const ctx = this.ctx;
    if (!ctx || !this.master) return;
    try { this.checkNode?.stop(); } catch { /* not started */ }
    const sr = ctx.sampleRate, n = Math.round(sr * 1.1);
    const buf = ctx.createBuffer(2, n, sr);
    const ch = buf.getChannelData(side === "left" ? 0 : 1);
    let lp = 0;
    for (let b = 0; b < 3; b++) {
      const start = Math.round(b * 0.36 * sr), len = Math.round(0.22 * sr);
      for (let i = 0; i < len; i++) {
        const env = Math.min(1, i / 400, (len - i) / 400);
        lp += 0.35 * ((Math.random() * 2 - 1) - lp);           // softened noise
        ch[start + i] = 0.35 * lp * env;
      }
    }
    // Silence the main mix (including the ambience tail) during the test so only one ear sounds.
    const bus = this.bus!;
    const t0 = ctx.currentTime;
    bus.gain.cancelScheduledValues(t0);
    bus.gain.setTargetAtTime(0, t0, 0.01);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.master);
    src.onended = () => {
      if (this.checkNode === src) {
        this.checkNode = null;
        bus.gain.setTargetAtTime(1, ctx.currentTime, 0.02);
      }
      onEnd();
    };
    src.start(t0 + 0.08);
    this.checkNode = src;
  }

  /** Dry (non-spatialised) preview of a sound through its own node. */
  preview(buffer: AudioBuffer, onEnd: () => void, seconds = 4): void {
    const ctx = this.ctx;
    if (!ctx) return;
    this.stopPreview();
    const src = ctx.createBufferSource();
    const g = ctx.createGain();
    g.gain.value = 0.5;
    src.buffer = buffer;
    src.connect(g).connect(ctx.destination);
    src.onended = () => { if (this.previewNode === src) this.previewNode = null; onEnd(); };
    src.start(0, 0, Math.min(seconds, buffer.duration));
    this.previewNode = src;
  }

  stopPreview() {
    try { this.previewNode?.stop(); } catch { /* not started */ }
    this.previewNode = null;
  }

  latencyInfo() {
    const ctx = this.ctx;
    if (!ctx) return null;
    const anyCtx = ctx as AudioContext & { outputLatency?: number };
    return {
      sampleRate: ctx.sampleRate,
      baseLatency: typeof ctx.baseLatency === "number" ? ctx.baseLatency : null,
      outputLatency: typeof anyCtx.outputLatency === "number" && anyCtx.outputLatency > 0 ? anyCtx.outputLatency : null,
    };
  }

  dispose() {
    if (this.pingTimer) window.clearInterval(this.pingTimer);
    void this.ctx?.close();
    this.ctx = null;
  }
}
