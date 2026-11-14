"""Export golden test vectors so the browser DSP can be verified against Python.

Writes to ``frontend/src/test/golden/``:
  * pack.ssfp     - the CIPIC filter pack (default subject), byte-identical to the API's
  * golden.json   - inputs and Python outputs for coordinates, fractional shift,
                    interpolation (all statuses), trajectory, limiter and full engine renders

Run:  python -m sonicsphere.export_golden
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .audio_processing import PeakLimiter
from .audio_renderer import ListenerPose, render_trajectory
from .config import get_settings
from .coordinate_system import (cart_to_sph, interaural_polar_to_cart, quantize_direction, rotate_about_up,
                                sph_to_cart, world_to_head)
from .filter_pack import get_pack, get_prepared
from .hrir_interpolation import HRIRInterpolator
from .hrir_preprocessing import frac_shift
from .trajectory import Trajectory

OUT = Path(__file__).resolve().parents[2] / "frontend" / "src" / "test" / "golden"


def _l(a, nd=9):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def main() -> int:
    s = get_settings()
    r = s.rendering
    cfg = s.dataset("cipic")
    prep = get_prepared("cipic", cfg.default_subject, None, s)
    pack, etag = get_pack("cipic", cfg.default_subject, None, s)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pack.ssfp").write_bytes(pack)
    pf = prep.filters
    rng = np.random.default_rng(2024)
    g: dict = {"etag": etag, "fs": pf.fs, "blockSize": r.block_size, "quantum": r.direction_quantum_deg}

    # Coordinates -------------------------------------------------------------
    coords = []
    for _ in range(40):
        src = rng.uniform(-3, 3, 3)
        pl = rng.uniform(-0.5, 0.5, 3) + [0, 1.1, 0]
        yaw, pitch = float(rng.uniform(-180, 180)), float(rng.uniform(-40, 40))
        h = world_to_head(src, pl, yaw, pitch)
        az, el, dist = cart_to_sph(h)
        coords.append({"src": _l(src), "listener": _l(pl), "yaw": yaw, "pitch": pitch, "head": _l(h, 12),
                       "az": float(az), "el": float(el), "r": float(dist),
                       "q": list(quantize_direction(float(az), float(el), r.direction_quantum_deg)),
                       "rot30": _l(rotate_about_up(h, 30.0), 12)})
    g["coords"] = coords

    # Fractional shift --------------------------------------------------------
    x = rng.standard_normal(256)
    g["fracshift"] = {"x": _l(x, 12), "nfft": 512,
                      "cases": [{"shift": sh, "y": _l(frac_shift(x, sh, 512), 12)} for sh in (0.0, 3.0, -7.25, 12.6)]}

    # Interpolation -----------------------------------------------------------
    it = HRIRInterpolator(pf, "barycentric_aligned", r.gap_edge_threshold_deg, r.measured_snap_deg,
                          triangulation=prep.triangulation, direction_quantum_deg=r.direction_quantum_deg)
    near = HRIRInterpolator(pf, "nearest")
    raw = HRIRInterpolator(pf, "barycentric_raw", r.gap_edge_threshold_deg, r.measured_snap_deg,
                           triangulation=prep.triangulation)
    dirs = [sph_to_cart(a, e) for a, e in ((0, 0), (37, 12), (90, 0), (271.3, -20), (180, 60), (0, -90),
                                           (45, -70), (123.4, 85))]
    dirs += [interaural_polar_to_cart(30, 270), interaural_polar_to_cart(-60, 250), pf.vectors[700]]
    v = rng.standard_normal((25, 3))
    dirs += list(v / np.linalg.norm(v, axis=1, keepdims=True))
    cases = []
    for d in dirs:
        q = it.query(d)
        cases.append({"dir": _l(d, 15), "status": q.w.status, "indices": [int(i) for i in q.w.indices],
                      "weights": _l(q.w.weights, 12), "toa": _l(q.toa, 9), "left": _l(q.hrir[0]),
                      "right": _l(q.hrir[1]), "nearest": int(near.query(d).w.indices[0]),
                      "rawLeft": _l(raw.query(d).hrir[0])})
    g["interp"] = cases

    # Trajectory --------------------------------------------------------------
    pts = np.array([[1.5, 1.2, 0.0], [0.3, 1.8, -1.4], [-1.4, 1.0, -0.2], [-0.2, 0.6, 1.5], [1.0, 1.5, 1.0]])
    tr_closed = Trajectory.from_points(pts, 3.0, loop=True)
    tr_open = Trajectory.from_points(pts, 3.0, loop=False)
    phases = [0.0, 0.1, 0.33, 0.5, 0.77, 0.999, 1.0]
    g["trajectory"] = {"points": _l(pts), "duration": 3.0,
                       "closed": {"samples": _l(tr_closed.samples, 12),
                                  "at": [_l(tr_closed.position_at_phase(p), 12) for p in phases]},
                       "open": {"samples": _l(tr_open.samples, 12),
                                "at": [_l(tr_open.position_at_phase(p), 12) for p in phases]},
                       "phases": phases}

    # Limiter -----------------------------------------------------------------
    xl = rng.standard_normal((3000, 2)) * np.linspace(0.2, 3.0, 3000)[:, None]
    lim = PeakLimiter(pf.fs, r.limiter_threshold, r.limiter_release_ms)
    g["limiter"] = {"x": _l(xl, 12), "y": _l(lim.process(xl), 12), "threshold": r.limiter_threshold,
                    "releaseMs": r.limiter_release_ms}

    # Engine ------------------------------------------------------------------
    n = int(0.4 * pf.fs)
    sig = 0.2 * rng.standard_normal(n)
    sig2 = 0.2 * rng.standard_normal(n)
    pose = ListenerPose(np.array([0.0, 1.15, 0.0]), 25.0, -5.0)
    eng_pts = np.array([[0.0, 1.15, -1.6], [-1.6, 1.6, 0.0], [0.0, 0.4, 1.6], [1.6, 1.0, 0.0]])
    traj = Trajectory.from_points(eng_pts, 0.35, loop=True)
    engine_cases = []
    for method in ("direct", "partitioned"):
        for voices in (1, 2):
            sigs = [sig] if voices == 1 else [sig, sig2]
            offs = [0.0] if voices == 1 else [15.0, -15.0]
            it._cached.cache_clear()
            y, eng = render_trajectory(sigs, offs, it, traj, pose, gain=pf.gain, method=method, tail=False,
                                       record_keys=True, limiter_threshold=r.limiter_threshold,
                                       limiter_release_ms=r.limiter_release_ms)
            engine_cases.append({"method": method, "voices": voices, "offsets": offs, "left": _l(y[:, 0]),
                                 "right": _l(y[:, 1]), "filterUpdates": eng.stats.filter_updates,
                                 "crossfades": eng.stats.crossfades,
                                 "keys": [list(k) for k in eng.stats.keys]})
    g["engine"] = {"signal": _l(sig, 12), "signal2": _l(sig2, 12), "pose": {"position": [0, 1.15, 0], "yaw": 25.0,
                                                                           "pitch": -5.0},
                   "points": _l(eng_pts), "duration": 0.35, "loop": True, "gain": pf.gain, "distanceAttenuation": True,
                   "cases": engine_cases}
    (OUT / "golden.json").write_text(json.dumps(g))
    print(f"Wrote {OUT / 'golden.json'} and {OUT / 'pack.ssfp'} ({len(pack) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
