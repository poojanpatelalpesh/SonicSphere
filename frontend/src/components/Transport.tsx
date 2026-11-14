import { Headphones, Pause, Play, Repeat, Square } from "lucide-react";
import { useStore } from "../state/store";
import { Meter, Panel, Seg, Slider, Toggle } from "./ui";

export function Transport() {
  const playing = useStore((s) => s.playing);
  const state = useStore((s) => s.audioState);
  const volume = useStore((s) => s.volume);
  const loop = useStore((s) => s.loopSound);
  const live = useStore((s) => s.live);
  const sel = useStore((s) => s.sounds.find((x) => x.id === s.selectedSoundId));
  const spatial = useStore((s) => s.spatial);
  const ambience = useStore((s) => s.ambience);
  const checking = useStore((s) => s.checking);
  const { setSpatial, setAmbience, checkHeadphones } = useStore.getState();
  const { play, pause, stop, setVolume, setLoopSound, startAudio } = useStore.getState();
  const pos = live?.transport.position ?? 0, dur = live?.transport.duration || sel?.durationS || 0;
  return (
    <Panel title="Listen" icon={<Headphones size={13} />} right={sel ? <span className="muted">{sel.name}</span> : null}>
      {(state === "off" || state === "error") && (
        <button className="btn btn-primary mb-3 w-full !py-2" onClick={() => void startAudio()}>
          <Headphones size={15} /> Put on headphones, then start sound
        </button>
      )}
      <div className="flex items-center gap-2">
        <button className="btn btn-primary !px-4 !py-2" onClick={() => (playing ? pause() : void play())} disabled={!sel}
                aria-label={playing ? "Pause" : "Play"}>
          {playing ? <Pause size={16} /> : <Play size={16} />}{playing ? "Pause" : "Play"}
        </button>
        <button className="btn !py-2" onClick={stop} aria-label="Stop"><Square size={14} />Stop</button>
        <button className="btn btn-icon !py-2" aria-pressed={loop} onClick={() => setLoopSound(!loop)} title={loop ? "Looping" : "Play once"}
                style={loop ? { borderColor: "var(--accent)", color: "var(--accent)" } : undefined}><Repeat size={15} /></button>
        <div className="ml-1 flex-1">
          <div className="h-1.5 overflow-hidden rounded-full" style={{ background: "var(--panel-2)" }}>
            <div className="h-full" style={{ width: `${dur ? (100 * pos) / dur : 0}%`, background: "var(--accent)" }} />
          </div>
          <div className="mt-1 flex justify-between font-mono text-[10.5px] muted"><span>{pos.toFixed(1)} s</span><span>{dur.toFixed(1)} s</span></div>
        </div>
      </div>
      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        <Slider id="volume" label="Volume" value={volume} min={0} max={1.5} step={0.01} onChange={setVolume} format={(v) => `${Math.round(v * 100)} %`} />
        <div className="space-y-1 pt-1">
          <Meter label="L" value={live?.peaks.left ?? 0} color="var(--left)" />
          <Meter label="R" value={live?.peaks.right ?? 0} color="var(--right)" />
        </div>
      </div>
      <div className="mt-3 rounded-xl border p-3" style={{ borderColor: spatial ? "var(--accent)" : "var(--line)" }}>
        <div className="flex items-center justify-between gap-2">
          <span className="text-[13px] font-semibold">3D sound</span>
          <Seg label="3D sound" value={spatial ? "on" : "off"} onChange={(v) => setSpatial(v === "on")}
               options={[{ id: "on", label: "On" }, { id: "off", label: "Off" }]} />
        </div>
        <p className="mt-1.5 text-[12px] muted">{spatial
          ? "Each ear gets its own filtered signal. Switch off to compare with the plain sound."
          : "Off: both ears hear the same plain sound (same loudness), so it sits inside your head. Switch on again."}</p>
        <div className="mt-2"><Toggle label="Room ambience" checked={ambience} onChange={setAmbience}
                                       hint="Adds soft reflections of a small room, which helps sounds appear outside your head" /></div>
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-2 text-[12px]">
        <span className="muted">Check your headphones:</span>
        <button className="btn !py-1 !text-xs" onClick={() => void checkHeadphones("left")}
                style={checking === "left" ? { borderColor: "var(--left)", color: "var(--left)" } : undefined}>Left ear only</button>
        <button className="btn !py-1 !text-xs" onClick={() => void checkHeadphones("right")}
                style={checking === "right" ? { borderColor: "var(--right)", color: "var(--right)" } : undefined}>Right ear only</button>
      </div>
      <p className="mt-1.5 text-[11.5px] muted">If both ears hear each test, your system is playing mono (turn off “Mono audio” / spatial-sound enhancements in your sound settings) – the 3D effect needs true stereo headphones.</p>
    </Panel>
  );
}
