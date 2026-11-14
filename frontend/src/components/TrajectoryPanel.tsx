import { Eraser, Hand, Pause, PenLine, Play, RotateCcw, Route, Spline, Square } from "lucide-react";
import { PRESETS } from "../lib/presets";
import { useStore, type Mode } from "../state/store";
import { Panel, Seg, Slider, Toggle } from "./ui";

export function TrajectoryPanel() {
  const mode = useStore((s) => s.mode);
  const traj = useStore((s) => s.traj);
  const playing = useStore((s) => s.trajPlaying);
    const phase = useStore((s) => s.live?.trajectory.phase ?? 0);
  const active = useStore((s) => s.live?.trajectory.active ?? false);
  const st = useStore.getState();
  let length = 0;
  if (traj.samples) for (let i = 3; i < traj.samples.length; i += 3) {
    length += Math.hypot(traj.samples[i] - traj.samples[i - 3], traj.samples[i + 1] - traj.samples[i - 2], traj.samples[i + 2] - traj.samples[i - 1]);
  }
  const modes: { id: Mode; label: React.ReactNode; title: string }[] = [
    { id: "position", label: <span className="flex items-center gap-1"><Hand size={12} />Position</span>, title: "Click the sphere to place the source, drag the orb to move it" },
    { id: "draw", label: <span className="flex items-center gap-1"><PenLine size={12} />Draw</span>, title: "Press and drag over the sphere to record a path" },
    { id: "edit", label: <span className="flex items-center gap-1"><Spline size={12} />Edit</span>, title: "Drag handles; click the sphere to insert; shift-click a handle to delete" },
  ];
  const hint = mode === "position" ? "Click anywhere on the sphere to put the sound there, or drag the glowing orb. Drag empty space to look around, scroll to zoom."
    : mode === "draw" ? "Press and drag across the sphere to draw a flight path for the sound. Release to finish. Turn the view first to draw behind the listener."
    : "Drag the purple handles to reshape the path. Click the sphere to add a point, shift-click a handle to remove it.";
  return (
    <Panel title="Move the sound" icon={<Route size={13} />} right={traj.samples ? <span className="muted">{traj.name || "path"}</span> : null}>
      <Seg label="Interaction mode" value={mode} options={modes} onChange={(m) => st.setMode(m)} />
      <p className="mt-2 text-[12px] muted">{hint}</p>
      <div className="mt-3 grid grid-cols-3 gap-1.5 sm:grid-cols-5">
        {PRESETS.map((p) => (
          <button key={p.id} className="btn !px-1.5 !text-xs" title={p.description}
                  onClick={() => { st.setTrajectoryDuration(p.duration); useStore.setState({ traj: { ...useStore.getState().traj, loop: p.loop } }); st.setTrajectory(p.points(useStore.getState().sphereRadius), p.name); }}>
            {p.name}
          </button>
        ))}
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <button className="btn btn-primary" disabled={!traj.samples} onClick={() => (playing ? st.pauseTrajectory() : void st.playTrajectory())}>
          {playing ? <Pause size={14} /> : <Play size={14} />}{playing ? "Pause path" : "Play path"}
        </button>
        <button className="btn" disabled={!traj.samples} onClick={() => st.stopTrajectory()}><Square size={13} />Stop</button>
        <button className="btn" disabled={!traj.samples} onClick={() => void st.replayTrajectory()}><RotateCcw size={13} />Replay</button>
        <button className="btn" disabled={!traj.samples} onClick={() => st.clearTrajectory()}><Eraser size={13} />Clear</button>
      </div>
      {traj.samples && (
        <>
          <div className="mt-3 h-1.5 overflow-hidden rounded-full" style={{ background: "var(--panel-2)" }}>
            <div className="h-full" style={{ width: `${active ? phase * 100 : 0}%`, background: "var(--accent-2)" }} />
          </div>
          <div className="mt-3 grid gap-3 sm:grid-cols-2">
            <Slider id="traj-dur" label="Time for one pass" value={traj.duration} min={1} max={30} step={0.5}
                    onChange={(v) => st.setTrajectoryDuration(v)} format={(v) => `${v.toFixed(1)} s`} />
            <div className="space-y-2 pt-1">
              <Toggle label="Loop" checked={traj.loop} onChange={(v) => st.setTrajectoryLoop(v)} hint="Repeat the path or fly it once" />
              <div className="text-[11.5px] muted">{length.toFixed(1)} m long · {(length / traj.duration).toFixed(1)} m/s</div>
            </div>
          </div>
        </>
      )}
    </Panel>
  );
}
