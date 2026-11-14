/** Plain-language helpers for the experiential UI. */
import { headToWorld, sphToCart } from "../dsp/coordinates";
import { HEAD, useStore } from "../state/store";

/** "front-left, slightly above" style description of a listener-relative direction. */
export function describeDirection(az: number, el: number): string {
  if (el > 70) return "directly above you";
  if (el < -70) return "directly below you";
  const a = ((az % 360) + 360) % 360;
  const names = ["in front of you", "front-left", "to your left", "behind-left", "behind you", "behind-right", "to your right", "front-right"];
  const side = names[Math.round(a / 45) % 8];
  const height = el > 35 ? ", high above" : el > 12 ? ", slightly above" : el < -35 ? ", far below" : el < -12 ? ", slightly below" : "";
  return side + height;
}

/** Place the source at a direction relative to the listener's current orientation. */
export function placeAt(azDeg: number, elDeg: number, distance: number): void {
  const s = useStore.getState();
  s.stopTrajectory();
  s.setSourcePos(headToWorld(sphToCart(azDeg, elDeg, distance), HEAD, s.listenerYaw, s.listenerPitch));
}

export const STATUS_TEXT: Record<string, string> = {
  measured: "using a real ear recording made at exactly this direction",
  interpolated: "blending the nearest real ear recordings around this direction",
  gap_fill: "no recordings exist here (below the head), so it is estimated from the nearest ones",
  clamped: "outside the recorded area, so the nearest recorded edge is used",
};
