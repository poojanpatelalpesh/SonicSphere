import { ArrowUpFromDot, Ear, Orbit, Plane, Sparkles, Undo2, Waves } from "lucide-react";
import { useState } from "react";
import { placeAt } from "../lib/describe";
import { PRESETS } from "../lib/presets";
import { useStore } from "../state/store";
import { Panel } from "./ui";

interface Experience {
  id: string;
  title: string;
  notice: string;
  icon: typeof Ear;
  run: () => Promise<void>;
}

async function playPreset(id: string) {
  const s = useStore.getState();
  const p = PRESETS.find((x) => x.id === id)!;
  s.setTrajectoryDuration(p.duration);
  useStore.setState({ traj: { ...useStore.getState().traj, loop: p.loop } });
  s.setTrajectory(p.points(s.sphereRadius), p.name);
  await s.replayTrajectory();
}

async function playStatic(az: number, el: number, r: number) {
  const s = useStore.getState();
  if (!(await s.startAudio())) return;
  placeAt(az, el, r);
  if (!useStore.getState().playing) await s.play();
}

const EXPERIENCES: Experience[] = [
  { id: "orbit", title: "Circle around your head", icon: Orbit, run: () => playPreset("orbit"),
    notice: "Follow it from front → left ear → behind you → right ear. Try it with your eyes closed." },
  { id: "leftear", title: "Whisper in your left ear", icon: Ear, run: () => playStatic(90, 0, 0.6),
    notice: "The biggest cues there are: it reaches the left ear first and much louder." },
  { id: "behind", title: "Right behind you", icon: Undo2, run: () => playStatic(180, 0, 1.5),
    notice: "Both ears hear it at the same time and level – only your ear shape tells front from back. Compare with “in front”." },
  { id: "front", title: "Straight in front", icon: Waves, run: () => playStatic(0, 0, 1.5),
    notice: "Same timing and level at both ears as “behind” – listen for the brighter, clearer colour." },
  { id: "above", title: "Above your head", icon: ArrowUpFromDot, run: () => playStatic(0, 75, 1.4),
    notice: "Height is heard mostly through subtle changes in tone made by your outer ear." },
  { id: "flyby", title: "Something flies past", icon: Plane, run: () => playPreset("flyby"),
    notice: "It sweeps from far front-left to your right, getting louder as it comes close." },
  { id: "spiral", title: "Spiral up and down", icon: Sparkles, run: () => playPreset("spiral"),
    notice: "Two turns around you while rising and falling." },
];

export function Experiences() {
  const [active, setActive] = useState<string | null>(null);
  const exp = EXPERIENCES.find((e) => e.id === active);
  return (
    <Panel title="Try this" icon={<Sparkles size={13} />} right={<span className="muted">use headphones</span>}>
      <div className="grid grid-cols-2 gap-1.5">
        {EXPERIENCES.map((e) => {
          const Icon = e.icon;
          const on = e.id === active;
          return (
            <button key={e.id} className="btn !justify-start !px-2.5 !py-2 text-left !text-[12.5px]"
                    style={on ? { borderColor: "var(--accent)", background: "color-mix(in srgb, var(--accent) 12%, var(--panel))" } : undefined}
                    onClick={() => { setActive(e.id); void e.run(); }}>
              <Icon size={15} style={{ color: "var(--accent)" }} className="shrink-0" />{e.title}
            </button>
          );
        })}
      </div>
      <p className="mt-3 min-h-[2.5em] text-[12.5px] leading-snug muted">
        {exp ? <><b style={{ color: "var(--text)" }}>What to notice: </b>{exp.notice}</> : "Pick an experience. The sound starts automatically and the orb shows where it is."}
      </p>
    </Panel>
  );
}
