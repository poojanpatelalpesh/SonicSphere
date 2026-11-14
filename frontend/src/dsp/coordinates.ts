/**
 * Coordinate systems – mirrors backend/sonicsphere/coordinate_system.py.
 *
 * World (Three.js scene): right-handed, +Y up, metres. At yaw 0 the listener faces −Z,
 *   so world −X is the listener's left.
 * Listener pose: head centre p_l, yaw ψ (about +Y, positive = turn left), pitch θ (nose up).
 * Head frame (SOFA): x = forward, y = left, z = up.
 * SOFA spherical: azimuth ∈ [0, 360) counter-clockwise from the front (90 = left),
 *   elevation ∈ [−90, 90].
 */

export type Vec3 = [number, number, number];

export const DEG = Math.PI / 180;
/** numpy.rad2deg multiplies by 180/pi; using the same constant keeps results bit-identical. */
export const RAD = 180 / Math.PI;

export function sphToCart(azDeg: number, elDeg: number, r = 1): Vec3 {
  const az = azDeg * DEG, el = elDeg * DEG;
  return [r * Math.cos(el) * Math.cos(az), r * Math.cos(el) * Math.sin(az), r * Math.sin(el)];
}

export function cartToSph(v: ArrayLike<number>): { az: number; el: number; r: number } {
  const x = v[0], y = v[1], z = v[2];
  const r = Math.sqrt(x * x + y * y + z * z);
  let az = (Math.atan2(y, x) * RAD) % 360;
  if (az < 0) az += 360;
  if (az >= 360) az -= 360;
  const el = Math.atan2(z, Math.hypot(x, y)) * RAD;
  return { az, el, r };
}

/** CIPIC-style interaural-polar angles (degrees). lateral = asin(y) (+left), polar = atan2(z, x) in [−90, 270). */
export function interauralPolar(v: ArrayLike<number>): { lateral: number; polar: number } {
  const n = Math.hypot(v[0], v[1], v[2]) || 1;
  const lateral = Math.asin(Math.max(-1, Math.min(1, v[1] / n))) * RAD;
  let polar = Math.atan2(v[2] / n, v[0] / n) * RAD;
  if (polar < -90) polar += 360;
  return { lateral, polar };
}

export function interauralPolarToCart(lateralDeg: number, polarDeg: number): Vec3 {
  const l = lateralDeg * DEG, p = polarDeg * DEG;
  return [Math.cos(l) * Math.cos(p), Math.sin(l), Math.cos(l) * Math.sin(p)];
}

/** Rows: listener forward, left, up axes in world coordinates. */
export function listenerBasis(yawDeg: number, pitchDeg = 0): [Vec3, Vec3, Vec3] {
  const psi = yawDeg * DEG, th = pitchDeg * DEG;
  const f: Vec3 = [-Math.sin(psi) * Math.cos(th), Math.sin(th), -Math.cos(psi) * Math.cos(th)];
  const l: Vec3 = [-Math.cos(psi), 0, Math.sin(psi)];
  const u: Vec3 = [f[1] * l[2] - f[2] * l[1], f[2] * l[0] - f[0] * l[2], f[0] * l[1] - f[1] * l[0]];
  return [f, l, u];
}

/** d = p_s − p_l expressed in the head frame (x fwd, y left, z up). */
export function worldToHead(src: ArrayLike<number>, listener: ArrayLike<number>, yawDeg: number, pitchDeg = 0,
                            out: Vec3 = [0, 0, 0]): Vec3 {
  const d0 = src[0] - listener[0], d1 = src[1] - listener[1], d2 = src[2] - listener[2];
  const [f, l, u] = listenerBasis(yawDeg, pitchDeg);
  out[0] = d0 * f[0] + d1 * f[1] + d2 * f[2];
  out[1] = d0 * l[0] + d1 * l[1] + d2 * l[2];
  out[2] = d0 * u[0] + d1 * u[1] + d2 * u[2];
  return out;
}

export function headToWorld(h: ArrayLike<number>, listener: ArrayLike<number>, yawDeg: number, pitchDeg = 0): Vec3 {
  const [f, l, u] = listenerBasis(yawDeg, pitchDeg);
  return [
    listener[0] + h[0] * f[0] + h[1] * l[0] + h[2] * u[0],
    listener[1] + h[0] * f[1] + h[1] * l[1] + h[2] * u[1],
    listener[2] + h[0] * f[2] + h[1] * l[2] + h[2] * u[2],
  ];
}

/** Rotate head-frame vector about the head's up axis (positive = towards the left). */
export function rotateAboutUp(v: ArrayLike<number>, angleDeg: number, out: Vec3 = [0, 0, 0]): Vec3 {
  const a = angleDeg * DEG, c = Math.cos(a), s = Math.sin(a);
  const x = v[0], y = v[1];
  out[0] = c * x - s * y;
  out[1] = s * x + c * y;
  out[2] = v[2];
  return out;
}

/** Great-circle angle (degrees) between two direction vectors – atan2(|a×b|, a·b). */
export function angularDistanceDeg(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const na = Math.hypot(a[0], a[1], a[2]), nb = Math.hypot(b[0], b[1], b[2]);
  const ax = a[0] / na, ay = a[1] / na, az = a[2] / na;
  const bx = b[0] / nb, by = b[1] / nb, bz = b[2] / nb;
  const cx = ay * bz - az * by, cy = az * bx - ax * bz, cz = ax * by - ay * bx;
  return Math.atan2(Math.hypot(cx, cy, cz), ax * bx + ay * by + az * bz) * RAD;
}

/** Filter-cache key grid (round half up, identical to the Python implementation). */
export function quantizeDirection(azDeg: number, elDeg: number, quantumDeg: number): [number, number] {
  const q = quantumDeg;
  const elK = Math.floor(elDeg / q + 0.5);
  const azK = Math.floor(azDeg / q + 0.5);
  let elQ = Math.min(90, Math.max(-90, elK * q));
  let azQ = azK * q;
  if (azQ >= 360 - 1e-9) azQ -= 360;
  if (Math.abs(elQ) >= 90 - 1e-9) azQ = 0;
  if (Object.is(elQ, -0)) elQ = 0;
  return [azQ, elQ];
}

export function normalize(v: Vec3): Vec3 {
  const n = Math.hypot(v[0], v[1], v[2]);
  return n > 0 ? [v[0] / n, v[1] / n, v[2] / n] : [1, 0, 0];
}
