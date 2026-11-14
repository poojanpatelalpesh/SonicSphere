import { Compass, RotateCcw } from "lucide-react";
import { cartToSph, worldToHead } from "../dsp/coordinates";
import { describeDirection, placeAt } from "../lib/describe";
import { HEAD, useStore } from "../state/store";
import { Panel, Slider } from "./ui";

export function ListenerPanel() {
  const yaw = useStore((s) => s.listenerYaw), pitch = useStore((s) => s.listenerPitch);
  const pos = useStore((s) => s.sourcePos);
  const st = useStore.getState();
  const sph = cartToSph(worldToHead(pos, HEAD, yaw, pitch));
  const setDistance = (r: number) => {
    st.setSphereRadius(r);
    if (!useStore.getState().live?.trajectory.active) placeAt(sph.az, sph.el, r);
  };
  return (
    <Panel title="Turn the listener" icon={<Compass size={13} />}>
      <p className="mb-3 text-[12.5px] muted">
        The sound stays where it is in the room – turn the listener and hear it move to the other side of the head, just like turning your own head.
      </p>
      <div className="grid gap-3 sm:grid-cols-2">
        <Slider id="yaw" label="Turn left / right" value={yaw} min={-180} max={180} step={1}
                onChange={(v) => st.setListener(v, pitch)} format={(v) => (v === 0 ? "ahead" : `${Math.abs(v).toFixed(0)}° ${v > 0 ? "left" : "right"}`)} />
        <Slider id="pitch" label="Look up / down" value={pitch} min={-40} max={40} step={1}
                onChange={(v) => st.setListener(yaw, v)} format={(v) => (v === 0 ? "level" : `${Math.abs(v).toFixed(0)}° ${v > 0 ? "up" : "down"}`)} />
        <Slider id="radius" label="Sound distance" value={Math.min(3, Math.max(0.6, Math.round(sph.r * 20) / 20))} min={0.6} max={3} step={0.05}
                onChange={setDistance} format={(v) => `${v.toFixed(1)} m`} />
        <div className="flex items-end">
          <button className="btn w-full" onClick={() => st.setListener(0, 0)}><RotateCcw size={13} />Face forward</button>
        </div>
      </div>
      <p className="mt-3 text-[12.5px]">Relative to the listener the sound is <b style={{ color: "var(--accent)" }}>{describeDirection(sph.az, sph.el)}</b>, {sph.r.toFixed(1)} m away.</p>
    </Panel>
  );
}
