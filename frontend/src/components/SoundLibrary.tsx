import { AudioLines, Crosshair, Guitar, Info, Mic, Plane, Play, Square, Trash2, Upload } from "lucide-react";
import { useRef, useState } from "react";
import { useStore, type Sound } from "../state/store";
import { Panel, Seg, Slider } from "./ui";

const ICONS: Record<string, typeof Guitar> = { helicopter: Plane, rifle: Crosshair, guitar: Guitar, voice: Mic, upload: AudioLines };

function Waveform({ peaks, active }: { peaks: Float32Array | null; active: boolean }) {
  if (!peaks) return <div className="h-8 w-full animate-pulse rounded" style={{ background: "var(--panel-2)" }} />;
  const n = peaks.length, w = 100 / n;
  return (
    <svg viewBox="0 0 100 32" preserveAspectRatio="none" className="h-8 w-full" aria-hidden>
      {Array.from(peaks).map((v, i) => {
        const h = Math.max(1, v * 30);
        return <rect key={i} x={i * w + w * 0.15} y={16 - h / 2} width={w * 0.7} height={h} rx={0.4}
                     fill={active ? "var(--accent)" : "var(--muted)"} opacity={active ? 0.95 : 0.5} />;
      })}
    </svg>
  );
}

function SoundCard({ s }: { s: Sound }) {
  const selected = useStore((st) => st.selectedSoundId === s.id);
  const previewing = useStore((st) => st.previewingId === s.id);
  const select = useStore((st) => st.selectSound);
  const preview = useStore((st) => st.previewSound);
  const remove = useStore((st) => st.removeSound);
  const Icon = ICONS[s.icon] ?? AudioLines;
  return (
    <div role="button" tabIndex={0} aria-pressed={selected} onClick={() => void select(s.id)}
         onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); void select(s.id); } }}
         className="group relative cursor-pointer rounded-xl border p-2.5 text-left transition focus-visible:outline focus-visible:outline-2"
         style={{ borderColor: selected ? "var(--accent)" : "var(--line)", background: selected ? "color-mix(in srgb, var(--accent) 9%, var(--panel))" : "var(--panel-2)",
                  boxShadow: selected ? "0 0 0 1px var(--accent)" : undefined }}>
      <div className="flex items-start gap-2">
        <div className="grid h-8 w-8 shrink-0 place-items-center rounded-lg" style={{ background: selected ? "var(--accent)" : "var(--panel)", color: selected ? "#031018" : "var(--accent)" }}>
          <Icon size={16} />
        </div>
        <div className="min-w-0 flex-1">
          <div className="truncate text-[13px] font-semibold" title={s.name}>{s.name}</div>
          <div className="text-[11px] muted">{s.durationS.toFixed(2)} s · {s.kind}{s.channelCount === 2 ? " · stereo" : ""}</div>
        </div>
        <button className="btn btn-icon !h-7 !w-7 !p-0" onClick={(e) => { e.stopPropagation(); void preview(s.id); }}
                aria-label={previewing ? `Stop preview of ${s.name}` : `Preview ${s.name} (dry, not spatialised)`}
                title={previewing ? "Stop preview" : "Preview (dry, not spatialised)"} disabled={!s.raw}>
          {previewing ? <Square size={12} /> : <Play size={12} />}
        </button>
      </div>
      <div className="mt-2"><Waveform peaks={s.peaks} active={selected} /></div>
      {s.error && <div className="mt-1 text-[11px]" style={{ color: "var(--danger)" }}>{s.error}</div>}
      <div className="mt-1 flex items-center gap-1 text-[10.5px] muted" title={`${s.source}\nLicense: ${s.license}`}>
        <Info size={11} /> <span className="truncate">{s.isUpload ? "Your upload" : s.license.split("(")[0]}</span>
        {s.isUpload && (
          <button className="ml-auto opacity-70 hover:opacity-100" aria-label={`Remove ${s.name}`}
                  onClick={(e) => { e.stopPropagation(); remove(s.id); }}><Trash2 size={12} /></button>
        )}
      </div>
    </div>
  );
}

function StereoChoice() {
  const policy = useStore((s) => s.stereoPolicy);
  const width = useStore((s) => s.stereoWidth);
  const setStereo = useStore((s) => s.setStereo);
  const prepared = useStore((s) => s.prepared);
  const sel = useStore((s) => s.sounds.find((x) => x.id === s.selectedSoundId));
  if ((sel?.channelCount ?? 1) !== 2) return null;
  const pair = policy === "stereo_pair";
  return (
    <div className="mt-3 rounded-xl border p-3 text-[12.5px]" style={{ borderColor: "var(--line)" }}>
      <div className="mb-2 font-semibold">Your file is stereo – how should it be placed?</div>
      <Seg label="Stereo handling" value={pair ? "stereo_pair" : "mid"} onChange={(v) => void setStereo(v)}
           options={[{ id: "mid", label: "One sound (mix L+R)" }, { id: "stereo_pair", label: "Two sounds side by side" }]} />
      {pair && <div className="mt-2"><Slider id="stereo-width" label="Distance between the two" value={width} min={10} max={120} step={5}
                                            onChange={(v) => void setStereo("stereo_pair", v)} format={(v) => `${v}°`} /></div>}
      <p className="mt-2 muted">{pair ? "The left channel becomes one sound source and the right channel another, both moving together."
        : prepared?.analysis.channels === 2 && (prepared.analysis.sideEnergyFraction ?? 0) > 0.05
          ? `Mixing removes the ${((prepared.analysis.sideEnergyFraction ?? 0) * 100).toFixed(0)} % of the sound that differs between left and right. Choose “two sounds” to keep it.`
          : "Both channels are combined into one sound source."}</p>
      {prepared?.analysis.warning && <p className="mt-1" style={{ color: "var(--warn)" }}>The two channels cancel when mixed – use “two sounds side by side”.</p>}
    </div>
  );
}

export function SoundLibrary() {
  const sounds = useStore((s) => s.sounds);
  const addUpload = useStore((s) => s.addUpload);
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  return (
    <Panel title="Choose a sound" icon={<AudioLines size={13} />} right={<span className="muted">{sounds.length} sounds</span>}>
      <div className="grid grid-cols-2 gap-2">
        {sounds.map((s) => <SoundCard key={s.id} s={s} />)}
        {sounds.length === 0 && <div className="col-span-2 py-6 text-center text-sm muted">Loading sounds…</div>}
      </div>
      <div onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
           onDrop={(e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files[0]; if (f) void addUpload(f); }}
           className="mt-2 flex items-center gap-3 rounded-xl border border-dashed p-3 text-[12.5px] transition"
           style={{ borderColor: over ? "var(--accent)" : "var(--line)", background: over ? "color-mix(in srgb, var(--accent) 8%, transparent)" : undefined }}>
        <Upload size={18} className="shrink-0" style={{ color: "var(--accent)" }} />
        <div className="flex-1">
          <div className="font-medium">Upload your own audio</div>
          <div className="muted">WAV or MP3 · up to 60 seconds · stays on your computer</div>
        </div>
        <button className="btn" onClick={() => input.current?.click()}>Choose file</button>
        <input ref={input} type="file" accept="audio/*,.wav,.mp3,.ogg,.flac" className="hidden"
               onChange={(e) => { const f = e.target.files?.[0]; if (f) void addUpload(f); e.target.value = ""; }} />
      </div>
      <StereoChoice />
    </Panel>
  );
}
