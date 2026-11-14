import { Moon, Sun, Volume2, VolumeX } from "lucide-react";
import { useStore } from "../state/store";

function AudioPill() {
  const state = useStore((s) => s.audioState);
  const playing = useStore((s) => s.playing);
  const start = useStore((s) => s.startAudio);
  const color = state === "running" ? (playing ? "var(--ok)" : "var(--accent)") : state === "error" ? "var(--danger)" : "var(--warn)";
  const text = state === "running" ? (playing ? "Playing" : "Sound ready") : state === "off" ? "Sound off" : state === "starting" ? "Starting…"
    : state === "error" ? "Sound error" : "Paused by browser – click to resume";
  return (
    <button className="chip !py-1 !text-xs" onClick={() => void start()} title="Click to start or resume sound">
      <span className={`h-2 w-2 rounded-full ${playing ? "pulse-dot" : ""}`} style={{ background: color }} />
      {state === "running" ? <Volume2 size={13} /> : <VolumeX size={13} />}
      <span style={{ color: "var(--text)" }}>{text}</span>
    </button>
  );
}

export function Header() {
  const theme = useStore((s) => s.theme);
  const setTheme = useStore((s) => s.setTheme);
  return (
    <header className="sticky top-0 z-30 border-b backdrop-blur" style={{ borderColor: "var(--line)", background: "color-mix(in srgb, var(--bg) 82%, transparent)" }}>
      <div className="mx-auto flex max-w-[1680px] items-center gap-4 px-4 py-2.5">
        <div className="flex items-center gap-2.5">
          <svg width="30" height="30" viewBox="0 0 64 64" aria-hidden>
            <defs><radialGradient id="lg" cx="50%" cy="45%" r="55%"><stop offset="0" stopColor="#67e8f9" /><stop offset="1" stopColor="#0e7490" /></radialGradient></defs>
            <circle cx="32" cy="32" r="28" fill="none" stroke="var(--accent)" strokeWidth="3" opacity=".45" />
            <ellipse cx="32" cy="32" rx="28" ry="10" fill="none" stroke="var(--accent)" strokeWidth="2" opacity=".7" />
            <ellipse cx="32" cy="32" rx="10" ry="28" fill="none" stroke="var(--accent-2)" strokeWidth="1.5" opacity=".5" />
            <circle cx="32" cy="32" r="10" fill="url(#lg)" />
          </svg>
          <div className="leading-tight">
            <div className="text-[15px] font-semibold tracking-tight">SonicSphere</div>
            <div className="text-[10.5px] muted">hear sound all around you – with ordinary headphones</div>
          </div>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <AudioPill />
          <button className="btn btn-icon" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
                  aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`} title="Toggle dark / light mode">
            {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
          </button>
        </div>
      </div>
    </header>
  );
}
