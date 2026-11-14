"""Source trajectories p_s(t) = [x(t), y(t), z(t)].

A trajectory is defined by control points (world coordinates). It is turned into a
smooth curve with a centripetal Catmull-Rom spline (alpha = 0.5, no cusps or
self-intersections within a segment), and resampled uniformly by arc length so that
the source moves at constant speed. Playback position = phase in [0, 1) times the
resampled polyline. Mirrors frontend/src/dsp/trajectory.ts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ALPHA = 0.5
SEGMENT_SUBDIVISIONS = 32
DEFAULT_SAMPLES = 512


def _catmull_rom_segment(p0, p1, p2, p3, n: int) -> np.ndarray:
    def knot(ti, a, b):
        return ti + max(float(np.linalg.norm(b - a)) ** ALPHA, 1e-4)

    t0 = 0.0
    t1 = knot(t0, p0, p1)
    t2 = knot(t1, p1, p2)
    t3 = knot(t2, p2, p3)
    out = np.empty((n, 3))
    for s in range(n):
        t = t1 + (t2 - t1) * s / n
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        out[s] = (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2
    return out


def spline_polyline(points: np.ndarray, closed: bool) -> np.ndarray:
    """Dense polyline through the control points (includes the final point / loop closure)."""
    p = np.asarray(points, dtype=float)
    k = p.shape[0]
    if k == 0:
        raise ValueError("trajectory needs at least one point")
    if k == 1:
        return p.copy()
    if k == 2 and not closed:
        return np.linspace(p[0], p[1], SEGMENT_SUBDIVISIONS + 1)
    segs = []
    n_seg = k if closed else k - 1
    for i in range(n_seg):
        if closed:
            q0, q1, q2, q3 = p[(i - 1) % k], p[i], p[(i + 1) % k], p[(i + 2) % k]
        else:
            q1, q2 = p[i], p[i + 1]
            q0 = p[i - 1] if i > 0 else 2 * p[0] - p[1]
            q3 = p[i + 2] if i + 2 < k else 2 * p[k - 1] - p[k - 2]
        segs.append(_catmull_rom_segment(q0, q1, q2, q3, SEGMENT_SUBDIVISIONS))
    segs.append((p[0] if closed else p[-1])[None, :])
    return np.concatenate(segs, axis=0)


def resample_by_arclength(poly: np.ndarray, n: int = DEFAULT_SAMPLES) -> np.ndarray:
    poly = np.asarray(poly, dtype=float)
    if poly.shape[0] == 1:
        return np.repeat(poly, n, axis=0)
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 1e-12:
        return np.repeat(poly[:1], n, axis=0)
    targets = np.linspace(0.0, total, n)
    out = np.empty((n, 3))
    j = 0
    for i, s in enumerate(targets):
        while j < len(seg) - 1 and cum[j + 1] < s:
            j += 1
        span = seg[j]
        u = 0.0 if span <= 1e-12 else (s - cum[j]) / span
        u = min(1.0, max(0.0, u))
        out[i] = poly[j] + u * (poly[j + 1] - poly[j])
    return out


@dataclass
class Trajectory:
    samples: np.ndarray   # (S, 3) uniformly spaced in arc length
    duration_s: float
    loop: bool = True

    @classmethod
    def from_points(cls, points: np.ndarray, duration_s: float, loop: bool = True, closed: bool | None = None,
                    n_samples: int = DEFAULT_SAMPLES) -> "Trajectory":
        if duration_s <= 0:
            raise ValueError("duration must be positive")
        closed = loop if closed is None else closed
        return cls(resample_by_arclength(spline_polyline(points, closed), n_samples), float(duration_s), loop)

    def phase_at(self, t_s: float) -> float:
        ph = t_s / self.duration_s
        if self.loop:
            return ph - np.floor(ph)
        return min(1.0, max(0.0, ph))

    def position_at_phase(self, phase: float) -> np.ndarray:
        s = self.samples
        u = min(1.0, max(0.0, phase)) * (len(s) - 1)
        i = int(np.floor(u))
        if i >= len(s) - 1:
            return s[-1].copy()
        f = u - i
        return s[i] + f * (s[i + 1] - s[i])

    def position_at(self, t_s: float) -> np.ndarray:
        return self.position_at_phase(self.phase_at(t_s))

    def length(self) -> float:
        return float(np.sum(np.linalg.norm(np.diff(self.samples, axis=0), axis=1)))
