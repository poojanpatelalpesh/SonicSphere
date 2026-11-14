/**
 * Application state (zustand) and the actions that drive the audio engine.
 *
 * High-rate data (worklet status at ~43 Hz) is kept in `liveRef` for the 3D scene's
 * render loop; the React store receives it throttled for the panels.
 */
import { create } from "zustand";
import { AudioEngine, type AudioState } from "../audio/AudioEngine";
import type { WorkletStatus } from "../audio/protocol";
import { runBenchmark, type BenchmarkReport } from "../dsp/benchmark";
import type { Vec3 } from "../dsp/coordinates";
import type { ConvMethod } from "../dsp/convolver";
import { parsePack, type FilterPack } from "../dsp/filterPack";
import type { InterpMethod } from "../dsp/interpolation";
import { prepareSource, type PreparedSource, type StereoPolicy } from "../dsp/sourcePrep";
import { Trajectory } from "../dsp/trajectory";
import { api, STATIC_DATA, type DatasetInfo, type DatasetSummary, type SoundEntry } from "../lib/api";

export const HEAD: Vec3 = [0, 1.15, 0];
/** Source loudness target (RMS); the output limiter prevents clipping. */
const SOURCE_RMS_DB = -16;
export const UPLOAD_MAX_BYTES = 25 * 1024 * 1024;
export const UPLOAD_MAX_SECONDS = 60;
export const UPLOAD_MIN_SECONDS = 0.01;
const UPLOAD_TYPES = ["wav", "wave", "mp3", "ogg", "oga", "flac", "m4a", "aac", "webm"];

export type Mode = "position" | "draw" | "edit";
export type MethodChoice = "auto" | ConvMethod;
export type Theme = "dark" | "light";

export interface Sound {
  id: string;
  name: string;
  kind: string;
  icon: string;
  description: string;
  source: string;
  license: string;
  durationS: number;
  sampleRate: number;
  url: string;
  channelCount: number;
  isUpload: boolean;
  raw: ArrayBuffer | null;
  file?: File;
  peaks: Float32Array | null;
  error?: string;
}

export interface TrajectoryState {
  points: Vec3[];
  samples: Float64Array | null;
  duration: number;
  loop: boolean;
  name: string;
}

export const liveRef: { status: WorkletStatus | null; receivedAt: number; ctxTime: number } = { status: null, receivedAt: 0, ctxTime: 0 };

interface Store {
  theme: Theme;
  backendError: string | null;
  datasets: DatasetSummary[];
  datasetId: string;
  subject: string;
  datasetInfo: DatasetInfo | null;
  pack: FilterPack | null;          // main-thread copy (visualisation, lab)
  packError: string | null;
  loadingPack: boolean;

  audioState: AudioState;
  audioError: string | null;
  sampleRate: number | null;
  baseLatency: number | null;
  outputLatency: number | null;
  controlRttMs: number | null;
  timerInWorklet: boolean;

  sounds: Sound[];
  selectedSoundId: string | null;
  loadedSoundId: string | null;
  prepared: PreparedSource | null;
  previewingId: string | null;
  playing: boolean;
  loopSound: boolean;
  volume: number;
  stereoPolicy: StereoPolicy;
  stereoWidth: number;

  mode: Mode;
  sourcePos: Vec3;
  sphereRadius: number;
  listenerYaw: number;
  listenerPitch: number;
  distanceAttenuation: boolean;
  crossfade: boolean;
  methodChoice: MethodChoice;
  method: ConvMethod;
  interp: InterpMethod;

  traj: TrajectoryState;
  trajPlaying: boolean;
  spatial: boolean;
  ambience: boolean;
  checking: "left" | "right" | null;

  live: WorkletStatus | null;
  bench: BenchmarkReport | null;
  benchRunning: boolean;
  toast: { kind: "info" | "error"; text: string } | null;

  // actions
  init(): Promise<void>;
  setTheme(t: Theme): void;
  selectDataset(id: string, subject?: string): Promise<void>;
  startAudio(): Promise<boolean>;
  selectSound(id: string): Promise<void>;
  addUpload(file: File): Promise<void>;
  removeSound(id: string): void;
  previewSound(id: string): Promise<void>;
  play(): Promise<void>;
  pause(): void;
  stop(): void;
  setVolume(v: number): void;
  setLoopSound(v: boolean): void;
  setStereo(policy: StereoPolicy, width?: number): Promise<void>;
  setMode(m: Mode): void;
  setSourcePos(p: Vec3): void;
  setSphereRadius(r: number): void;
  setListener(yaw: number, pitch: number): void;
  setDistanceAttenuation(v: boolean): void;
  setCrossfade(v: boolean): void;
  setMethodChoice(m: MethodChoice): void;
  setInterp(m: InterpMethod): void;
  setTrajectory(points: Vec3[], name?: string): void;
  setTrajectoryDuration(d: number): void;
  setTrajectoryLoop(v: boolean): void;
  clearTrajectory(): void;
  playTrajectory(): Promise<void>;
  pauseTrajectory(): void;
  stopTrajectory(): void;
  replayTrajectory(): Promise<void>;
  runBench(): void;
  setSpatial(v: boolean): void;
  setAmbience(v: boolean): void;
  checkHeadphones(side: "left" | "right"): Promise<void>;
  notify(kind: "info" | "error", text: string): void;
}

let engine: AudioEngine | null = null;
let lastUiUpdate = 0;
export const getEngine = () => engine;

function computePeaks(ch: Float32Array, bins = 96): Float32Array {
  const out = new Float32Array(bins);
  const step = Math.max(1, Math.floor(ch.length / bins));
  for (let b = 0; b < bins; b++) {
    let m = 0;
    for (let i = b * step; i < Math.min(ch.length, (b + 1) * step); i++) m = Math.max(m, Math.abs(ch[i]));
    out[b] = m;
  }
  const mx = Math.max(...out) || 1;
  return out.map((v) => v / mx);
}

async function decodeOffline(raw: ArrayBuffer): Promise<AudioBuffer> {
  const off = new OfflineAudioContext(1, 1, 44100);
  return off.decodeAudioData(raw.slice(0));
}

function channelsOf(buf: AudioBuffer): Float32Array[] {
  return Array.from({ length: buf.numberOfChannels }, (_, i) => buf.getChannelData(i).slice());
}

export const useStore = create<Store>((set, get) => ({
  theme: (() => { try { return localStorage.getItem("sonicsphere-theme") === "light" ? "light" : "dark"; } catch { return "dark"; } })(),
  backendError: null,
  datasets: [],
  datasetId: "cipic",
  subject: "165",
  datasetInfo: null,
  pack: null,
  packError: null,
  loadingPack: false,

  audioState: "off",
  audioError: null,
  sampleRate: null,
  baseLatency: null,
  outputLatency: null,
  controlRttMs: null,
  timerInWorklet: false,

  sounds: [],
  selectedSoundId: null,
  loadedSoundId: null,
  prepared: null,
  previewingId: null,
  playing: false,
  loopSound: true,
  volume: 1,
  stereoPolicy: "mid",
  stereoWidth: 30,

  mode: "position",
  sourcePos: [-1.13, 1.15, -1.13],
  sphereRadius: 1.6,
  listenerYaw: 0,
  listenerPitch: 0,
  distanceAttenuation: true,
  crossfade: true,
  methodChoice: "auto",
  method: "partitioned",
  interp: "barycentric_aligned",

  traj: { points: [], samples: null, duration: 6, loop: true, name: "" },
  trajPlaying: false,
  spatial: true,
  ambience: true,
  checking: null,

  live: null,
  bench: null,
  benchRunning: false,
  toast: null,

  notify(kind, text) {
    set({ toast: { kind, text } });
    window.setTimeout(() => { if (get().toast?.text === text) set({ toast: null }); }, kind === "error" ? 7000 : 3500);
  },

  setTheme(t) {
    document.documentElement.classList.toggle("dark", t === "dark");
    try { localStorage.setItem("sonicsphere-theme", t); } catch { /* storage unavailable */ }
    set({ theme: t });
  },

  async init() {
    get().setTheme(get().theme);
    try {
      const [datasets, sounds] = await Promise.all([api.datasets(), api.sounds()]);
      set({ datasets, backendError: null });
      const ds = datasets.find((d) => d.id === "cipic" && d.downloaded_subjects.length) ?? datasets.find((d) => d.downloaded_subjects.length);
      if (!ds) {
        set({ backendError: STATIC_DATA ? "The ear-filter data is missing from this site build." : "No HRIR dataset is downloaded. Run: python -m sonicsphere.download_dataset" });
      } else {
        const subj = ds.downloaded_subjects.includes(ds.default_subject) ? ds.default_subject : ds.downloaded_subjects[0];
        await get().selectDataset(ds.id, subj);
      }
      const list: Sound[] = sounds.map((s: SoundEntry) => ({
        id: s.id, name: s.name, kind: s.kind, icon: s.icon, description: s.description, source: s.source, license: s.license,
        durationS: s.duration_s, sampleRate: s.sample_rate, url: s.url, channelCount: s.channels, isUpload: false, raw: null, peaks: null,
      }));
      set({ sounds: list, selectedSoundId: list[0]?.id ?? null });
      await Promise.all(list.map(async (s) => {
        try {
          const raw = await api.soundData(s.url);
          const buf = await decodeOffline(raw);
          set({ sounds: get().sounds.map((x) => (x.id === s.id ? { ...x, raw, peaks: computePeaks(buf.getChannelData(0)) } : x)) });
        } catch (e) {
          set({ sounds: get().sounds.map((x) => (x.id === s.id ? { ...x, error: (e as Error).message } : x)) });
        }
      }));
    } catch (e) {
      set({ backendError: (e as Error).message });
    }
  },

  async selectDataset(id, subject) {
    const ds = get().datasets.find((d) => d.id === id);
    const subj = subject ?? ds?.default_subject ?? "";
    set({ loadingPack: true, packError: null, datasetId: id, subject: subj });
    try {
      const info = await api.dataset(id, subj);
      const fs = engine?.ctx ? engine.ctx.sampleRate : info.sample_rate;
      const buf = await api.filters(id, subj, fs !== info.sample_rate ? fs : undefined);
      const pack = parsePack(buf.slice(0));
      set({ datasetInfo: info, pack, loadingPack: false });
      if (engine?.ctx) engine.sendPack(buf, get().method, get().interp);
    } catch (e) {
      set({ packError: (e as Error).message, loadingPack: false });
    }
  },

  async startAudio() {
    const st = get();
    if (engine?.ctx && st.audioState === "running") return true;
    if (!st.datasetInfo) { get().notify("error", "No dataset loaded – is the backend running?"); return false; }
    try {
      if (!engine) {
        engine = new AudioEngine({
          onState: (s) => {
            const info = engine?.latencyInfo();
            set({ audioState: s, sampleRate: info?.sampleRate ?? null, baseLatency: info?.baseLatency ?? null,
                  outputLatency: info?.outputLatency ?? null });
          },
          onStatus: (s) => {
            liveRef.status = s;
            liveRef.receivedAt = performance.now();
            liveRef.ctxTime = s.time;
            const t = performance.now();
            if (t - lastUiUpdate > 70) {
              lastUiUpdate = t;
              const info = engine?.latencyInfo();
              set({ live: s, playing: s.transport.playing, trajPlaying: s.trajectory.playing,
                    outputLatency: info?.outputLatency ?? get().outputLatency });
            }
          },
          onLatency: (ms) => set({ controlRttMs: ms }),
          onEnded: () => set({ playing: false }),
          onTrajectoryEnded: () => set({ trajPlaying: false }),
          onError: (m) => { set({ audioError: m }); get().notify("error", m); },
        });
      }
      const ctx = await engine.start(st.datasetInfo.sample_rate);
      set({ timerInWorklet: engine.timerInWorklet });
      // Filters at the context's actual rate (the backend resamples if the browser refused 44.1 kHz).
      const buf = await api.filters(st.datasetId, st.subject, ctx.sampleRate !== st.datasetInfo.sample_rate ? ctx.sampleRate : undefined);
      const pack = parsePack(buf.slice(0));
      set({ pack });
      // No start-up benchmark: it blocked the page for seconds on slow laptops. Measurements on
      // V8 showed the partitioned FFT path is 2.7–9x faster than direct, so it is the default.
      engine.sendPack(buf, get().method, get().interp);
      const s = get();
      engine.setListener(HEAD, s.listenerYaw, s.listenerPitch);
      engine.setPosition(s.sourcePos);
      engine.settings({ volume: s.volume, crossfade: s.crossfade, distanceAttenuation: s.distanceAttenuation, loop: s.loopSound, spatial: s.spatial });
      engine.setAmbience(s.ambience);
      if (s.traj.samples) engine.setTrajectory(s.traj.samples, s.traj.duration, s.traj.loop);
      set({ timerInWorklet: engine.timerInWorklet, audioError: null });
      return true;
    } catch (e) {
      set({ audioState: "error", audioError: (e as Error).message });
      get().notify("error", `Audio could not start: ${(e as Error).message}`);
      return false;
    }
  },

  async selectSound(id) {
    set({ selectedSoundId: id });
    if (!engine?.ctx) return;
    const snd = get().sounds.find((s) => s.id === id);
    if (!snd?.raw) return;
    try {
      const buf = await engine.ctx.decodeAudioData(snd.raw.slice(0));
      const prep = prepareSource(channelsOf(buf), get().stereoPolicy, get().stereoWidth, SOURCE_RMS_DB, -1);
      engine.setSource(id, prep.signals, prep.offsetsDeg, get().loopSound);
      set({ loadedSoundId: id, prepared: prep });
      if (prep.silent) get().notify("info", "This sound is silent – nothing will be heard.");
    } catch (e) {
      get().notify("error", `Could not decode ${snd.name}: ${(e as Error).message}`);
    }
  },

  async addUpload(file) {
    const ext = file.name.split(".").pop()?.toLowerCase() ?? "";
    if (!UPLOAD_TYPES.includes(ext) && !file.type.startsWith("audio/")) {
      get().notify("error", `Unsupported file type ".${ext}". Use WAV or MP3 (OGG/FLAC where the browser supports them).`);
      return;
    }
    if (file.size > UPLOAD_MAX_BYTES) { get().notify("error", `File is ${(file.size / 1e6).toFixed(1)} MB; the limit is 25 MB.`); return; }
    if (file.size === 0) { get().notify("error", "The file is empty."); return; }
    let raw: ArrayBuffer, buf: AudioBuffer;
    try {
      raw = await file.arrayBuffer();
      buf = await decodeOffline(raw);
    } catch {
      get().notify("error", `“${file.name}” could not be decoded. The file may be corrupt or use an unsupported codec.`);
      return;
    }
    if (buf.duration > UPLOAD_MAX_SECONDS) { get().notify("error", `Audio is ${buf.duration.toFixed(1)} s long; the limit is 60 s.`); return; }
    if (buf.duration < UPLOAD_MIN_SECONDS) { get().notify("error", "Audio is shorter than 10 ms."); return; }
    if (buf.numberOfChannels > 2) { get().notify("error", `${buf.numberOfChannels}-channel audio is not supported; use mono or stereo.`); return; }
    const id = `upload-${Date.now()}`;
    const snd: Sound = {
      id, name: file.name.replace(/\.[^.]+$/, ""), kind: "upload", icon: "upload", description: `${buf.numberOfChannels === 2 ? "Stereo" : "Mono"} upload`,
      source: "Uploaded by you (stays in this browser; sent to the backend only for a reference render)", license: "Your file",
      durationS: buf.duration, sampleRate: buf.sampleRate, url: "", channelCount: buf.numberOfChannels, isUpload: true, raw, file,
      peaks: computePeaks(buf.getChannelData(0)),
    };
    set({ sounds: [...get().sounds, snd] });
    await get().selectSound(id);
    get().notify("info", `Loaded “${snd.name}” (${buf.duration.toFixed(2)} s, ${buf.numberOfChannels === 2 ? "stereo" : "mono"}).`);
  },

  removeSound(id) {
    const s = get();
    set({ sounds: s.sounds.filter((x) => x.id !== id), selectedSoundId: s.selectedSoundId === id ? s.sounds[0]?.id ?? null : s.selectedSoundId });
  },

  async previewSound(id) {
    if (get().previewingId === id) { engine?.stopPreview(); set({ previewingId: null }); return; }
    if (!(await get().startAudio())) return;
    const snd = get().sounds.find((s) => s.id === id);
    if (!snd?.raw || !engine?.ctx) return;
    const buf = await engine.ctx.decodeAudioData(snd.raw.slice(0));
    if (get().playing) get().pause();
    set({ previewingId: id });
    engine.preview(buf, () => set({ previewingId: null }));
  },

  async play() {
    if (!(await get().startAudio())) return;
    const s = get();
    engine?.stopPreview();
    set({ previewingId: null });
    if (!s.selectedSoundId) { get().notify("info", "Pick a sound first."); return; }
    if (s.loadedSoundId !== s.selectedSoundId || !s.prepared) await get().selectSound(s.selectedSoundId);
    engine?.transport("play");
    set({ playing: true });
  },
  pause() { engine?.transport("pause"); set({ playing: false }); },
  stop() { engine?.transport("stop"); set({ playing: false }); },
  setVolume(v) { engine?.settings({ volume: v }); set({ volume: v }); },
  setLoopSound(v) { engine?.settings({ loop: v }); set({ loopSound: v }); },
  async setStereo(policy, width) {
    set({ stereoPolicy: policy, stereoWidth: width ?? get().stereoWidth });
    const id = get().loadedSoundId;
    if (id) await get().selectSound(id);
  },

  setMode(m) { set({ mode: m }); },
  setSourcePos(p) {
    set({ sourcePos: p });
    engine?.setPosition(p);
  },
  setSphereRadius(r) { set({ sphereRadius: r }); },
  setListener(yaw, pitch) {
    set({ listenerYaw: yaw, listenerPitch: pitch });
    engine?.setListener(HEAD, yaw, pitch);
  },
  setDistanceAttenuation(v) { engine?.settings({ distanceAttenuation: v }); set({ distanceAttenuation: v }); },
  setCrossfade(v) { engine?.settings({ crossfade: v }); set({ crossfade: v }); },
  setMethodChoice(m) {
    const method: ConvMethod = m === "auto" ? get().bench?.recommended ?? "partitioned" : m;
    engine?.settings({ method });
    set({ methodChoice: m, method });
  },
  setInterp(m) { engine?.settings({ interp: m }); set({ interp: m }); },

  setTrajectory(points, name) {
    const t = get().traj;
    if (points.length === 0) { get().clearTrajectory(); return; }
    const tr = Trajectory.fromPoints(points, t.duration, t.loop);
    set({ traj: { ...t, points, samples: tr.samples, name: name ?? t.name } });
    engine?.setTrajectory(tr.samples, t.duration, t.loop);
  },
  setTrajectoryDuration(d) {
    set({ traj: { ...get().traj, duration: d } });
    engine?.setTrajectoryDuration(d);
  },
  setTrajectoryLoop(v) {
    const t = get().traj;
    set({ traj: { ...t, loop: v } });
    if (t.points.length) get().setTrajectory(t.points);
  },
  clearTrajectory() {
    engine?.trajectoryTransport("stop");
    engine?.setTrajectory(null, 1, true);
    set({ traj: { ...get().traj, points: [], samples: null, name: "" }, trajPlaying: false });
  },
  async playTrajectory() {
    if (!get().traj.samples) { get().notify("info", "Draw or choose a trajectory first."); return; }
    if (!(await get().startAudio())) return;
    if (!get().playing) await get().play();
    engine?.trajectoryTransport("play");
    set({ trajPlaying: true, mode: "position" });
  },
  pauseTrajectory() { engine?.trajectoryTransport("pause"); set({ trajPlaying: false }); },
  stopTrajectory() {
    engine?.trajectoryTransport("stop");
    set({ trajPlaying: false });
    engine?.setPosition(get().sourcePos);
  },
  async replayTrajectory() {
    if (!(await get().startAudio())) return;
    if (!get().playing) await get().play();
    engine?.trajectoryTransport("restart");
    set({ trajPlaying: true });
  },

  setSpatial(v) { engine?.settings({ spatial: v }); set({ spatial: v }); },
  setAmbience(v) { engine?.setAmbience(v); set({ ambience: v }); },
  async checkHeadphones(side) {
    if (!(await get().startAudio())) return;
    if (get().playing) get().pause();
    set({ checking: side });
    engine?.headphoneCheck(side, () => { if (get().checking === side) set({ checking: null }); });
  },

  runBench() {
    const pack = get().pack;
    if (!pack) return;
    set({ benchRunning: true });
    window.setTimeout(() => {
      const bench = runBenchmark(pack, 600);
      const m = get().methodChoice;
      set({ bench, benchRunning: false, method: m === "auto" ? bench.recommended : get().method });
      if (m === "auto") engine?.settings({ method: bench.recommended });
    }, 30);
  },
}));
