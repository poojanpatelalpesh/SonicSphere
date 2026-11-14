/** Trajectory presets, defined in listener-relative directions (yaw 0) and mapped to world space. */
import { sphToCart, type Vec3 } from "../dsp/coordinates";
import { HEAD } from "../state/store";

/** Head-frame (x fwd, y left, z up) → world for the default listener orientation. */
export function headDirToWorld(az: number, el: number, r: number): Vec3 {
  const h = sphToCart(az, el, r);
  return [HEAD[0] - h[1], HEAD[1] + h[2], HEAD[2] - h[0]];
}

export interface Preset { id: string; name: string; description: string; duration: number; loop: boolean; points: (r: number) => Vec3[] }

const range = (n: number) => Array.from({ length: n }, (_, i) => i);

export const PRESETS: Preset[] = [
  { id: "orbit", name: "Orbit", description: "Horizontal circle at ear height (counter-clockwise: front → left → back → right)",
    duration: 8, loop: true, points: (r) => range(12).map((i) => headDirToWorld(i * 30, 0, r)) },
  { id: "spiral", name: "Spiral", description: "Two turns rising to +60° and descending to −35°",
    duration: 12, loop: true, points: (r) => range(24).map((i) => headDirToWorld(i * 30, 47.5 * Math.sin((i / 24) * 2 * Math.PI) + 12.5, r)) },
  { id: "overhead", name: "Over the head", description: "Median-plane circle: front → above → behind → below (crosses the unmeasured gap)",
    duration: 8, loop: true, points: (r) => range(12).map((i) => {
      const p = i * 30; // polar angle in the median plane
      return headDirToWorld(p <= 90 || p >= 270 ? 0 : 180, p <= 90 ? p : p >= 270 ? p - 360 : 180 - p, r);
    }) },
  { id: "figure8", name: "Figure eight", description: "Sweeps 110° left and right through the front, looping up and down",
    duration: 10, loop: true, points: (r) => range(16).map((i) => {
      const t = (i / 16) * 2 * Math.PI;
      return headDirToWorld((110 * Math.sin(t) + 360) % 360, 35 * Math.sin(2 * t), r);
    }) },
  { id: "flyby", name: "Fly-by", description: "Straight pass from far front-left to the right, 1.95 m from the head at the closest point",
    duration: 6, loop: false, points: () => [-5, -1.8, 1.8, 5].map((x) => [x, HEAD[1] + 0.3, 0.8 * x - 2.5] as Vec3) },
];
