/**
 * Experiential explanation of binaural rendering: the three cues the brain uses,
 * shown live for the current source direction, plus the two actual ear signals.
 */
import { Ear, Headphones, Music2, Shuffle } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { cartToSph, worldToHead } from "../dsp/coordinates";
import { RealFFT } from "../dsp/fft";
import { HRIRInterpolator } from "../dsp/interpolation";
import { Trajectory } from "../dsp/trajectory";
import { describeDirection, STATUS_TEXT } from "../lib/describe";
import { getEngine, HEAD, liveRef, useStore } from "../state/store";
import { LineChart } from "./charts";
import { Panel } from "./ui";

interface Cues {
  az: number; el: number; status: string; neighbours: number;
  itdUs: number; ildDb: number; freq: Float64Array; magL: Float64Array; magR: Float64Array;
}

const NFFT = 512;

function magnitude(h: ArrayLike<number>, fft: RealFFT): Float64Array {
  const re = new Float64Array(NFFT / 2 + 1), im = new Float64Array(NFFT / 2 + 1);
  fft.forward(h, re, im, Math.min(h.length, NFFT));
  // Light 1/6-octave-ish smoothing so the curve shows the broad colouring, not every ripple.
  const db = re.map((r, k) => 20 * Math.log10(Math.max(Math.hypot(r, im[k]), 1e-6)));
  return db.map((_, k) => {
    const w = Math.max(1, Math.round(k * 0.06));
    let s = 0, n = 0;
    for (let j = Math.max(0, k - w); j <= Math.min(db.length - 1, k + w); j++) { s += db[j]; n++; }
    return s / n;
  });
}

/** Cues of the filter pair currently in use (live from the audio engine, or computed from the same data). */
function useEarCues(): Cues | null {
  const pack = useStore((s) => s.pack);
  const audio = useStore((s) => s.audioState);
  const interp = useStore((s) => s.interp);
  const [cues, setCues] = useState<Cues | null>(null);
  const local = useMemo(() => (pack ? new HRIRInterpolator(pack, interp) : null), [pack, interp]);
  const fft = useMemo(() => new RealFFT(NFFT), []);
  useEffect(() => {
    if (!pack || !local) return;
    let alive = true;
    const fs = pack.header.fs, L = pack.L;
    const build = (filter: Float32Array, toa: [number, number], az: number, el: number, status: string, neighbours: number) => {
      let eL = 0, eR = 0;
      for (let i = 0; i < L; i++) { eL += filter[i] * filter[i]; eR += filter[L + i] * filter[L + i]; }
      setCues({
        az, el, status, neighbours,
        itdUs: ((toa[1] - toa[0]) / fs) * 1e6, ildDb: 10 * Math.log10(eL / eR),
        freq: Float64Array.from({ length: NFFT / 2 + 1 }, (_, k) => (k * fs) / NFFT),
        magL: magnitude(filter.subarray(0, L), fft), magR: magnitude(filter.subarray(L), fft),
      });
    };
    const tick = async () => {
      const eng = getEngine();
      if (audio === "running" && eng && liveRef.status) {
        const s = await eng.snapshot();
        if (alive && s.status) build(s.filter, s.toa, s.status.az, s.status.el, s.status.status, s.status.indices.length);
        return;
      }
      const st = useStore.getState();
      let pos = st.sourcePos;
      if (st.traj.samples && liveRef.status?.trajectory.active) {
        pos = new Trajectory(st.traj.samples, st.traj.duration, st.traj.loop).positionAtPhase(liveRef.status.trajectory.phase);
      }
      const r = local.query(worldToHead(pos, HEAD, st.listenerYaw, st.listenerPitch));
      if (alive) build(r.filter, r.toa, r.az, r.el, r.w.status, r.w.n);
    };
    void tick();
    const id = window.setInterval(() => void tick(), 300);
    return () => { alive = false; window.clearInterval(id); };
  }, [pack, local, audio, fft]);
  return cues;
}

/** Top-down head with the incoming sound and the ear that hears it first. */
function TimingView({ c }: { c: Cues }) {
  const a = (c.az * Math.PI) / 180;
  const R = 74, cx = 100, cy = 92;
  const sx = cx - Math.sin(a) * R, sy = cy - Math.cos(a) * R;
  const first = Math.abs(c.itdUs) < 25 ? null : c.itdUs > 0 ? "left" : "right";
  const glow = (ear: "left" | "right") => (first === ear ? (ear === "left" ? "var(--left)" : "var(--right)") : "var(--line)");
  return (
    <svg viewBox="0 0 200 184" className="mx-auto h-40 w-full max-w-[220px]" role="img" aria-label="Top view of the head with the incoming sound">
      <circle cx={cx} cy={cy} r={R} fill="none" stroke="var(--line)" strokeDasharray="3 4" />
      <text x={cx} y={cy - R - 6} textAnchor="middle" fontSize="9" fill="var(--muted)">FRONT</text>
      {/* wavefronts from the source */}
      {[0.35, 0.6, 0.85].map((f) => (
        <circle key={f} cx={sx} cy={sy} r={R * f} fill="none" stroke="var(--warn)" strokeOpacity={0.5 - f * 0.4} />
      ))}
      <line x1={sx} y1={sy} x2={cx} y2={cy} stroke="var(--warn)" strokeWidth="1.5" strokeDasharray="4 3" opacity="0.7" />
      <ellipse cx={cx} cy={cy} rx="24" ry="28" fill="var(--panel-2)" stroke="var(--muted)" />
      <path d={`M${cx - 6} ${cy - 27} L${cx} ${cy - 36} L${cx + 6} ${cy - 27}`} fill="var(--panel-2)" stroke="var(--muted)" />
      <ellipse cx={cx - 26} cy={cy} rx="5" ry="9" fill={glow("left")} />
      <ellipse cx={cx + 26} cy={cy} rx="5" ry="9" fill={glow("right")} />
      <text x={cx - 40} y={cy + 4} textAnchor="end" fontSize="10" fill="var(--left)">L</text>
      <text x={cx + 40} y={cy + 4} fontSize="10" fill="var(--right)">R</text>
      <circle cx={sx} cy={sy} r="7" fill="var(--warn)" />
    </svg>
  );
}

function LevelBars({ c }: { c: Cues }) {
  // Bars on a 30 dB scale relative to the louder ear.
  const l = c.ildDb >= 0 ? 0 : c.ildDb, r = c.ildDb >= 0 ? -c.ildDb : 0;
  const h = (db: number) => Math.max(4, 100 * (1 + db / 30));
  return (
    <div className="flex h-40 items-end justify-center gap-8" role="img" aria-label="Loudness at each ear">
      {([["L", l, "var(--left)"], ["R", r, "var(--right)"]] as const).map(([n, db, col]) => (
        <div key={n} className="flex h-full w-14 flex-col items-center justify-end">
          <span className="mb-1 font-mono text-[11px] muted">{db === 0 ? "loudest" : `${db.toFixed(1)} dB`}</span>
          <div className="w-full rounded-t-lg transition-[height] duration-200" style={{ height: `${h(db) * 0.75}%`, background: col, opacity: 0.85 }} />
          <span className="mt-1 text-xs font-semibold" style={{ color: col }}>{n} ear</span>
        </div>
      ))}
    </div>
  );
}

/** The two signals actually leaving the renderer (shared scale, so the level difference is visible). */
function EarWaves() {
  const ref = useRef<HTMLCanvasElement>(null);
  const audio = useStore((s) => s.audioState);
  const theme = useStore((s) => s.theme);
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const g = cv.getContext("2d")!;
    const css = getComputedStyle(document.documentElement);
    const cols = [css.getPropertyValue("--left").trim(), css.getPropertyValue("--right").trim()];
    const line = css.getPropertyValue("--line").trim(), muted = css.getPropertyValue("--muted").trim();
    const N = 1024, bl = new Float32Array(2048), br = new Float32Array(2048);
    let gain = 1, raf = 0, visible = true;
    // Only animate while sound plays and the canvas is on screen (saves CPU/GPU on laptops).
    const io = new IntersectionObserver(([e]) => { visible = e.isIntersecting; });
    io.observe(cv);
    const draw = () => {
      if (!visible || !useStore.getState().playing) { raf = window.setTimeout(() => requestAnimationFrame(draw), 250) as unknown as number; return; }
      const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
      if (cv.width !== w * dpr) { cv.width = w * dpr; cv.height = h * dpr; }
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, w, h);
      const eng = getEngine();
      const rows: [Float32Array, string, string][] = [[bl, cols[0], "left ear"], [br, cols[1], "right ear"]];
      if (eng?.analyserL && eng.analyserR) { eng.analyserL.getFloatTimeDomainData(bl); eng.analyserR.getFloatTimeDomainData(br); }
      let pk = 0;
      for (let i = 2048 - N; i < 2048; i++) pk = Math.max(pk, Math.abs(bl[i]), Math.abs(br[i]));
      const target = Math.min(60, 0.9 / Math.max(pk, 1e-3));
      gain = target < gain ? target : gain + 0.04 * (target - gain);
      rows.forEach(([buf, col, label], r) => {
        const y0 = (r + 0.5) * (h / 2), amp = (h / 2) * 0.44;
        g.strokeStyle = line; g.lineWidth = 1; g.beginPath(); g.moveTo(0, y0); g.lineTo(w, y0); g.stroke();
        g.fillStyle = muted; g.font = "10px Inter, sans-serif"; g.fillText(label, 6, r * (h / 2) + 12);
        g.strokeStyle = col; g.lineWidth = 1.4; g.beginPath();
        for (let i = 0; i < N; i++) {
          const x = (i / (N - 1)) * w, y = y0 - Math.max(-1, Math.min(1, buf[2048 - N + i] * gain)) * amp;
          if (i) g.lineTo(x, y); else g.moveTo(x, y);
        }
        g.stroke();
      });
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => { cancelAnimationFrame(raf); window.clearTimeout(raf); io.disconnect(); };
  }, [audio, theme]);
  return <canvas ref={ref} className="h-28 w-full rounded-xl" style={{ background: "var(--panel-2)" }} aria-label="Live left and right ear signals" />;
}

function Cue({ n, title, value, children, text }: { n: number; title: string; value: string; children: React.ReactNode; text: string }) {
  return (
    <div className="rounded-xl border p-3" style={{ borderColor: "var(--line)", background: "var(--panel-2)" }}>
      <div className="flex items-baseline justify-between gap-2">
        <div className="text-[13px] font-semibold"><span className="mr-1.5 font-mono" style={{ color: "var(--accent)" }}>{n}</span>{title}</div>
        <div className="font-mono text-[12px] tabular-nums" style={{ color: "var(--text)" }}>{value}</div>
      </div>
      <div className="my-1">{children}</div>
      <p className="text-[12px] leading-snug muted">{text}</p>
    </div>
  );
}

export function EarCues() {
  const c = useEarCues();
  const playing = useStore((s) => s.playing);
  if (!c) return <Panel title="What your ears are hearing"><p className="text-sm muted">Loading the ear recordings…</p></Panel>;
  const itd = Math.abs(c.itdUs), ild = Math.abs(c.ildDb);
  const firstEar = c.itdUs > 0 ? "left" : "right";
  const louderEar = c.ildDb > 0 ? "left" : "right";
  const fs = c.freq[c.freq.length - 1] * 2;
  return (
    <Panel title="What your ears are hearing" icon={<Ear size={13} />} right={<span className="muted">updates as the sound moves</span>}>
      <p className="mb-3 text-[15px]">
        The sound is <b style={{ color: "var(--accent)" }}>{describeDirection(c.az, c.el)}</b>
        <span className="text-[12.5px] muted"> · {c.az.toFixed(0)}° around, {c.el.toFixed(0)}° up · {STATUS_TEXT[c.status] ?? ""}</span>
      </p>
      <div className="grid gap-3 md:grid-cols-3">
        <Cue n={1} title="Arrival time" value={itd < 25 ? "same time" : `${(itd / 1000).toFixed(2)} ms`}
             text={itd < 25 ? "It reaches both ears together – it is in front, behind, above or below you."
               : `It reaches your ${firstEar} ear first. The sound has to travel ${(itd * 0.0343).toFixed(1)} cm further around your head to the other ear. Your brain detects differences as small as 0.01 ms.`}>
          <TimingView c={c} />
        </Cue>
        <Cue n={2} title="Loudness" value={ild < 1 ? "equal" : `${ild.toFixed(1)} dB`}
             text={ild < 1 ? "Both ears get about the same level." : `Your head shades the far ear, so the ${louderEar} ear is louder – especially for high sounds, which cannot bend around the head.`}>
          <LevelBars c={c} />
        </Cue>
        <Cue n={3} title="Ear-shape colouring" value="tone per ear"
             text="Folds of your outer ear boost and cut certain high pitches differently for every direction. That colouring is how you tell front from back and up from down.">
          <LineChart ariaLabel="How each ear colours the sound" height={160} xLog xDomain={[200, fs / 2]} yDomain={[-35, 20]} legend={false}
                     xLabel="pitch (Hz) →" yLabel="boost / cut (dB)"
                     series={[{ name: "left", color: "var(--left)", x: c.freq, y: c.magL, width: 2 }, { name: "right", color: "var(--right)", x: c.freq, y: c.magR, width: 2 }]} />
        </Cue>
      </div>
      <div className="mt-3">
        <div className="mb-1 flex items-center justify-between text-[12px]">
          <span className="font-semibold">The two signals sent to your headphones</span>
          <span className="muted">{playing ? "live · about 23 ms shown" : "press Play to see them"}</span>
        </div>
        <EarWaves />
        <p className="mt-1 text-[12px] muted">Same sound, two different versions – shifted in time, different in loudness and in tone. Headphones deliver each version to one ear only, so your brain places the sound out in space.</p>
      </div>
    </Panel>
  );
}

const STEPS = [
  { icon: Music2, title: "1 · A plain sound", text: "Any recording – a helicopter, a voice, your own file. On its own it has no direction." },
  { icon: Ear, title: "2 · Real ear recordings", text: "Researchers put tiny microphones in the ears of people and of a dummy head and played test sounds from 1250 directions. Each recording captures how the head and ears change a sound coming from that spot." },
  { icon: Shuffle, title: "3 · Filter for each ear", text: "For the orb’s direction the two matching recordings (blended from the nearest ones when needed) are applied to the sound – one for the left ear, one for the right." },
  { icon: Headphones, title: "4 · Your brain does the rest", text: "Headphones keep the two signals apart. Your brain reads the timing, loudness and tone differences – exactly as with a real sound – and hears it out there." },
];

export function HowItWorks() {
  return (
    <Panel title="How binaural rendering works" icon={<Headphones size={13} />}>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {STEPS.map((s, i) => {
          const Icon = s.icon;
          return (
            <div key={s.title} className="relative rounded-xl border p-3" style={{ borderColor: "var(--line)" }}>
              <div className="mb-2 grid h-9 w-9 place-items-center rounded-lg" style={{ background: "color-mix(in srgb, var(--accent) 15%, transparent)", color: "var(--accent)" }}>
                <Icon size={18} />
              </div>
              <div className="text-[13.5px] font-semibold">{s.title}</div>
              <p className="mt-1 text-[12.5px] leading-snug muted">{s.text}</p>
              {i < STEPS.length - 1 && <span className="absolute -right-2.5 top-1/2 hidden -translate-y-1/2 text-lg lg:block" style={{ color: "var(--accent)" }}>›</span>}
            </div>
          );
        })}
      </div>
      <p className="mt-3 text-[12px] muted">
        Why it may not be perfect for you: everyone’s ears are different, and these recordings come from someone else’s (or a dummy head’s) ears.
        If “behind” sounds like “in front”, that is normal – moving sounds and turning the listener help a lot.
      </p>
    </Panel>
  );
}

/** Shared readout for the scene overlay. */
export function useSceneDirection() {
  const pos = useStore((s) => s.sourcePos), yaw = useStore((s) => s.listenerYaw), pitch = useStore((s) => s.listenerPitch);
  const live = useStore((s) => s.live);
  const v = live?.voices[0];
  const h = cartToSph(worldToHead(pos, HEAD, yaw, pitch));
  const traj = !!live?.trajectory.active;
  return { az: v && traj ? v.az : h.az, el: v && traj ? v.el : h.el, traj };
}
