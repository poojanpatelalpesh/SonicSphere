/**
 * Source preparation for uploads and built-in sounds – mirrors audio_processing.py.
 *
 * Stereo policy (always shown to the user, never silent):
 *   mid          s = (L + R) / 2; the analysis reports the side-signal energy this removes
 *   left / right one channel
 *   stereo_pair  L and R become two virtual sources at ± width/2 around the source direction
 */

export type StereoPolicy = "mid" | "left" | "right" | "stereo_pair";

export const STEREO_POLICIES: { id: StereoPolicy; label: string; hint: string }[] = [
  { id: "mid", label: "Mono downmix (L+R)/2", hint: "One point source. Side (L−R) content is removed and reported." },
  { id: "stereo_pair", label: "Stereo pair", hint: "L and R rendered as two sources ±width/2 around the source. Nothing discarded." },
  { id: "left", label: "Left channel", hint: "Use only the left channel." },
  { id: "right", label: "Right channel", hint: "Use only the right channel." },
];

export interface StereoAnalysis {
  channels: number;
  correlation?: number;
  sideEnergyFraction?: number;
  identicalChannels?: boolean;
  warning?: string | null;
}

export interface PreparedSource {
  signals: Float32Array[];
  offsetsDeg: number[];
  policy: StereoPolicy | "mono";
  analysis: StereoAnalysis;
  normalisationGain: number;
  peak: number;
  rms: number;
  silent: boolean;
}

export function analyseStereo(ch: Float32Array[]): StereoAnalysis {
  if (ch.length === 1) return { channels: 1 };
  const [l, r] = ch;
  let eMid = 0, eSide = 0, ll = 0, rr = 0, lr = 0, same = true;
  for (let i = 0; i < l.length; i++) {
    const m = 0.5 * (l[i] + r[i]), s = 0.5 * (l[i] - r[i]);
    eMid += m * m; eSide += s * s; ll += l[i] * l[i]; rr += r[i] * r[i]; lr += l[i] * r[i];
    if (same && Math.abs(l[i] - r[i]) > 1e-6) same = false;
  }
  const tot = eMid + eSide;
  const den = Math.sqrt(ll * rr);
  const corr = den > 0 ? lr / den : 1;
  return {
    channels: 2,
    correlation: corr,
    sideEnergyFraction: tot > 0 ? eSide / tot : 0,
    identicalChannels: same,
    warning: corr < -0.5 ? "Channels are strongly anti-correlated; a mid downmix would cancel. Use 'stereo pair', 'left' or 'right'." : null,
  };
}

export function prepareSource(channels: Float32Array[], policy: StereoPolicy, widthDeg: number,
                              targetRmsDb = -20, peakCeilingDb = -1): PreparedSource {
  if (channels.length === 0 || channels.length > 2) throw new Error(`${channels.length}-channel audio is not supported`);
  const analysis = analyseStereo(channels);
  let signals: Float32Array[];
  let offsets: number[];
  let used: PreparedSource["policy"] = policy;
  if (channels.length === 1) {
    signals = [Float32Array.from(channels[0])]; offsets = [0]; used = "mono";
  } else if (policy === "mid") {
    const [l, r] = channels;
    const m = new Float32Array(l.length);
    for (let i = 0; i < l.length; i++) m[i] = 0.5 * (l[i] + r[i]);
    signals = [m]; offsets = [0];
  } else if (policy === "left") {
    signals = [Float32Array.from(channels[0])]; offsets = [0];
  } else if (policy === "right") {
    signals = [Float32Array.from(channels[1])]; offsets = [0];
  } else {
    signals = [Float32Array.from(channels[0]), Float32Array.from(channels[1])];
    offsets = [widthDeg / 2, -widthDeg / 2];
  }
  // One common gain: reach the RMS target unless that pushes the peak above the ceiling.
  let ss = 0, n = 0, peak = 0;
  for (const s of signals) for (let i = 0; i < s.length; i++) { ss += s[i] * s[i]; n++; const a = Math.abs(s[i]); if (a > peak) peak = a; }
  const rms = n ? Math.sqrt(ss / n) : 0;
  let g = 1;
  const silent = peak <= 1e-9;
  if (!silent) {
    g = Math.pow(10, targetRmsDb / 20) / Math.max(rms, 1e-12);
    g = Math.min(g, Math.pow(10, peakCeilingDb / 20) / peak);
    for (const s of signals) for (let i = 0; i < s.length; i++) s[i] *= g;
  }
  return { signals, offsetsDeg: offsets, policy: used, analysis, normalisationGain: g, peak, rms, silent };
}
