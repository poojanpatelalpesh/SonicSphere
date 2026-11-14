/** Message protocol between the main thread (AudioEngine) and the AudioWorklet. */
import type { Vec3 } from "../dsp/coordinates";
import type { ConvMethod } from "../dsp/convolver";
import type { DirStatus, InterpMethod } from "../dsp/interpolation";

export type ToWorklet =
  | { type: "pack"; pack: ArrayBuffer; method: ConvMethod; interp: InterpMethod }
  | { type: "source"; id: string; signals: Float32Array[]; offsets: number[]; loop: boolean }
  | { type: "transport"; action: "play" | "pause" | "stop" }
  | { type: "position"; pos: Vec3; seq: number; t: number }
  | { type: "listener"; position: Vec3; yaw: number; pitch: number }
  | { type: "trajectory"; samples: Float64Array | null; duration: number; loop: boolean }
  | { type: "trajectoryTransport"; action: "play" | "pause" | "stop" | "restart" }
  | { type: "trajectoryDuration"; duration: number }
  | { type: "settings"; volume?: number; spatial?: boolean; method?: ConvMethod; interp?: InterpMethod; crossfade?: boolean; distanceAttenuation?: boolean; loop?: boolean }
  | { type: "snapshot" }
  | { type: "ping"; seq: number; t: number };

export interface VoiceStatus {
  status: DirStatus;
  az: number;            // quantised requested direction (deg)
  el: number;
  renderedAz: number;
  renderedEl: number;
  indices: number[];
  weights: number[];
  toa: [number, number];
  nearestIndex: number;
  nearestAngleDeg: number;
  maxEdgeDeg: number;
  lateralDeg?: number;
  polarLoDeg?: number;
  polarHiDeg?: number;
}

export interface WorkletStatus {
  type: "status";
  frame: number;
  time: number;
  sampleRate: number;
  blockSize: number;
  filterLength: number;
  transport: { playing: boolean; position: number; duration: number; sourceId: string | null; loop: boolean };
  trajectory: { active: boolean; playing: boolean; phase: number; duration: number; loop: boolean };
  position: Vec3;
  distance: number;
  headDir: Vec3;
  voices: VoiceStatus[];
  method: ConvMethod;
  interp: InterpMethod;
  spatial: boolean;
  stats: { blocks: number; filterUpdates: number; crossfades: number; cacheHits: number; cacheMisses: number; cacheSize: number };
  timing: { available: boolean; estimated: boolean; meanUs: number; maxUs: number; overruns: number; windowBlocks: number;
            load: number; economy: boolean; quantum: number };
  limiterMinGain: number;
  peaks: { left: number; right: number; input: number };
}

export type FromWorklet =
  | { type: "ready"; sampleRate: number; blockSize: number; timer: boolean }
  | WorkletStatus
  | { type: "ack"; seq: number; t: number; frame: number }
  | { type: "pong"; seq: number; t: number; frame: number }
  | { type: "snapshot"; filter: Float32Array; toa: [number, number]; L: number; status: VoiceStatus | null }
  | { type: "ended" }
  | { type: "trajectoryEnded" }
  | { type: "error"; message: string };
