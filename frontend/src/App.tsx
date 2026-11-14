import { AlertTriangle, Headphones, X } from "lucide-react";
import { lazy, Suspense, useEffect, useState } from "react";
import { Experiences } from "./components/Experiences";
import { Header } from "./components/Header";
import { EarCues, HowItWorks, useSceneDirection } from "./components/HowItWorks";
import { ListenerPanel } from "./components/ListenerPanel";
import { SoundLibrary } from "./components/SoundLibrary";
import { TrajectoryPanel } from "./components/TrajectoryPanel";
import { Transport } from "./components/Transport";
import { describeDirection } from "./lib/describe";
import { STATIC_DATA } from "./lib/api";
import { useStore } from "./state/store";

const Scene = lazy(() => import("./scene/Scene").then((m) => ({ default: m.Scene })));

function SceneOverlay() {
  const { az, el, traj } = useSceneDirection();
  const mode = useStore((s) => s.mode);
  const chip = { background: "color-mix(in srgb, var(--panel) 70%, transparent)" };
  return (
    <>
      <div className="pointer-events-none absolute left-3 top-3 rounded-xl border px-3 py-2 backdrop-blur"
           style={{ borderColor: "var(--line)", background: "color-mix(in srgb, var(--panel) 78%, transparent)" }}>
        <div className="text-[10px] font-semibold uppercase tracking-[.14em] muted">{traj ? "sound is flying" : "sound is"}</div>
        <div className="text-[14px] font-semibold" style={{ color: "var(--accent)" }}>{describeDirection(az, el)}</div>
      </div>
      <div className="pointer-events-none absolute bottom-3 left-3 flex flex-wrap gap-2 text-[11px]">
        <span className="chip backdrop-blur" style={chip}>
          {mode === "position" ? "click or drag to move the sound" : mode === "draw" ? "drag on the sphere to draw a path" : "drag the handles to edit the path"}
        </span>
        <span className="chip backdrop-blur" style={chip}>
          <span className="h-2 w-2 rounded-full" style={{ background: "var(--left)" }} />left ear
          <span className="ml-1 h-2 w-2 rounded-full" style={{ background: "var(--right)" }} />right ear
        </span>
        <span className="chip backdrop-blur" style={chip}>dots = directions where real ear recordings were made</span>
      </div>
    </>
  );
}

export default function App() {
  const init = useStore((s) => s.init);
  const backendError = useStore((s) => s.backendError);
  const toast = useStore((s) => s.toast);
  const audioState = useStore((s) => s.audioState);
  const [hint, setHint] = useState(true);
  useEffect(() => { void init(); }, [init]);
  return (
    <div className="min-h-full">
      <Header />
      {backendError && (
        <div className="mx-auto mt-3 flex max-w-[1680px] items-start gap-2 rounded-xl border px-4 py-3 text-sm" style={{ borderColor: "var(--danger)", color: "var(--danger)" }} role="alert">
          <AlertTriangle size={16} className="mt-0.5 shrink-0" />
          <div><b>{STATIC_DATA ? "Could not load the sound data:" : "The sound server is not reachable:"}</b> {backendError}
            {!STATIC_DATA && <div className="muted">Start it with <code>uvicorn sonicsphere.main:app --port 8000</code> in <code>backend/</code>, then reload.</div>}</div>
        </div>
      )}
      {hint && audioState === "off" && !backendError && (
        <div className="mx-auto mt-3 flex max-w-[1680px] items-center gap-2 px-4 text-[13px] muted">
          <Headphones size={16} style={{ color: "var(--accent)" }} />
          Put on headphones and pick something under “Try this” – or choose a sound, press Play and drag the glowing orb around the listener.
          <button className="ml-auto opacity-60 hover:opacity-100" aria-label="Dismiss" onClick={() => setHint(false)}><X size={14} /></button>
        </div>
      )}
      <main className="mx-auto grid max-w-[1680px] gap-4 px-4 py-4 xl:grid-cols-[minmax(0,1fr)_420px]">
        <div className="min-w-0 space-y-4">
          <div className="panel relative h-[60vh] min-h-[420px] overflow-hidden !p-0">
            <Suspense fallback={<div className="grid h-full place-items-center text-sm muted">Loading 3D scene…</div>}><Scene /></Suspense>
            <SceneOverlay />
          </div>
          <EarCues />
          <HowItWorks />
        </div>
        <aside className="space-y-4">
          <Transport />
          <Experiences />
          <SoundLibrary />
          <TrajectoryPanel />
          <ListenerPanel />
        </aside>
      </main>
      <footer className="mx-auto max-w-[1680px] px-4 pb-6 pt-2 text-[11.5px] muted">
        Ear recordings: CIPIC HRTF Database © The Regents of the University of California. Built-in sounds: see each sound card (ⓘ) for its source and licence.
      </footer>
      {toast && (
        <div role="status" className="fixed bottom-4 right-4 z-50 max-w-sm rounded-xl border px-4 py-3 text-sm shadow-lg"
             style={{ background: "var(--panel)", borderColor: toast.kind === "error" ? "var(--danger)" : "var(--accent)" }}>
          {toast.text}
        </div>
      )}
    </div>
  );
}
