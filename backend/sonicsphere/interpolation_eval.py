"""Held-out evaluation of HRIR interpolation methods.

For each subject and hold-out fraction, a random subset of measured directions is
withheld (vertices on the boundary of the measured region are never withheld, since
estimating them would be extrapolation). Every method then estimates the withheld
HRIRs from the remaining measurements only, and the estimates are compared with the
true measurements:

* NMSE      time-domain normalised squared error, dB
* LSD       log-spectral distortion, 200 Hz - 16 kHz, dB (RMS of the dB magnitude error)
* TOA err   |onset(estimate) - onset(truth)|, microseconds, averaged over both ears
* ITD err   |ITD(estimate) - ITD(truth)|, microseconds (ITD from onset differences)
* cost      wall-clock time per direction query, microseconds

Vertex consistency
    NMSE between each method's representation evaluated with all weight on one
    measurement and the measured HRIR itself. Measured HRIRs are used verbatim at
    measured directions, so a method whose estimates do not converge to them (e.g.
    minimum phase, which discards the measured excess phase) produces a filter jump
    whenever a moving source passes a measurement point.

Selection rule (documented, applied automatically): among methods whose mean ITD
error is below ``ITD_LIMIT_US``, whose 95th-percentile vertex-consistency error is
below ``VERTEX_LIMIT_DB`` and whose mean query cost is below ``COST_LIMIT_US``,
choose the lowest mean LSD. (Costs are for the Python reference implementation; the
real-time cost of the chosen method is measured separately in the browser.)

Run:  python -m sonicsphere.interpolation_eval [--subjects 003 165] [--fractions 0.1 0.25]
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from .config import get_settings
from .dataset_loader import get_dataset
from .hrir_interpolation import METHODS, HRIRInterpolator, build_triangulation
from .hrir_preprocessing import PreparedFilters, estimate_toa, prepare_filters, reconstruct

ITD_LIMIT_US = 20.0
VERTEX_LIMIT_DB = -30.0
COST_LIMIT_US = 5000.0
LSD_BAND_HZ = (200.0, 16000.0)


def _subset(pf: PreparedFilters, keep: np.ndarray) -> PreparedFilters:
    return PreparedFilters(fs=pf.fs, filter_length=pf.filter_length, margin=pf.margin,
                           measured=pf.measured[keep], aligned=pf.aligned[keep], toa=pf.toa[keep],
                           minphase=pf.minphase[keep], vectors=pf.vectors[keep], gain=pf.gain,
                           truncated_energy_db=pf.truncated_energy_db, source_fs=pf.source_fs,
                           source_length=pf.source_length)


def lsd_db(est: np.ndarray, ref: np.ndarray, fs: float, nfft: int = 1024) -> np.ndarray:
    E = np.abs(np.fft.rfft(est, nfft, axis=-1))
    R = np.abs(np.fft.rfft(ref, nfft, axis=-1))
    f = np.fft.rfftfreq(nfft, 1 / fs)
    band = (f >= LSD_BAND_HZ[0]) & (f <= min(LSD_BAND_HZ[1], fs / 2))
    d = 20 * np.log10(np.maximum(E[..., band], 1e-12) / np.maximum(R[..., band], 1e-12))
    return np.sqrt(np.mean(d ** 2, axis=-1))


def nmse_db(est: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return 10 * np.log10(np.sum((est - ref) ** 2, axis=-1) / np.maximum(np.sum(ref ** 2, axis=-1), 1e-30))


def vertex_consistency(pf: PreparedFilters) -> dict[str, dict[str, float]]:
    """NMSE (dB) of each representation, all weight on one vertex, vs the measured HRIR."""
    reps = {
        "raw": pf.measured,
        "aligned": reconstruct(pf.aligned, pf.toa, pf.margin, pf.filter_length),
        "minphase": reconstruct(pf.minphase, pf.toa, 0, pf.filter_length),
    }
    out = {}
    for m, (_, rep) in METHODS.items():
        e = np.maximum(nmse_db(reps[rep], pf.measured), -300.0)
        out[m] = {"vertex_nmse_db_median": float(np.median(e)), "vertex_nmse_db_p95": float(np.percentile(e, 95))}
    return out


def holdout_indices(pf: PreparedFilters, fraction: float, seed: int, gap_threshold: float) -> np.ndarray:
    tri = build_triangulation(pf.vectors, gap_threshold)
    protected = set(np.unique(tri.boundary_edges).tolist())
    # Vertices of large triangles (pole caps) are also protected: their neighbourhood is sparse.
    for t in np.nonzero(tri.max_edge_deg > 0.6 * gap_threshold)[0]:
        protected.update(tri.triangles[t].tolist())
    cand = np.array([i for i in range(len(pf.vectors)) if i not in protected])
    rng = np.random.default_rng(seed)
    n = max(1, int(round(fraction * len(pf.vectors))))
    return np.sort(rng.choice(cand, size=min(n, len(cand)), replace=False))


def evaluate_subject(pf: PreparedFilters, fraction: float, seed: int, settings, methods=None) -> dict:
    r = settings.rendering
    held = holdout_indices(pf, fraction, seed, r.gap_edge_threshold_deg)
    keep = np.setdiff1d(np.arange(len(pf.vectors)), held)
    train = _subset(pf, keep)
    tri = build_triangulation(train.vectors, r.gap_edge_threshold_deg)
    truth = pf.measured[held]
    true_toa = pf.toa[held]
    out = {}
    vc = vertex_consistency(pf)
    for m in methods or METHODS:
        interp = HRIRInterpolator(train, m, r.gap_edge_threshold_deg, measured_snap_deg=0.0,
                                  triangulation=tri if METHODS[m][0] == "barycentric" else None)
        est = np.empty_like(truth)
        statuses = []
        t0 = time.perf_counter()
        for j, i in enumerate(held):
            res = interp.query(pf.vectors[i])
            est[j] = res.hrir
            statuses.append(res.w.status)
        cost = (time.perf_counter() - t0) / len(held) * 1e6
        est_toa = estimate_toa(est, r.onset_threshold_db, r.onset_upsample)
        toa_err = np.abs(est_toa - true_toa) / pf.fs * 1e6
        itd_err = np.abs((est_toa[:, 1] - est_toa[:, 0]) - (true_toa[:, 1] - true_toa[:, 0])) / pf.fs * 1e6
        out[m] = {
            "nmse_db": float(np.mean(nmse_db(est, truth))),
            "lsd_db": float(np.mean(lsd_db(est, truth, pf.fs))),
            "lsd_db_p95": float(np.percentile(lsd_db(est, truth, pf.fs), 95)),
            "toa_err_us": float(np.mean(toa_err)),
            "itd_err_us": float(np.mean(itd_err)),
            "itd_err_us_p95": float(np.percentile(itd_err, 95)),
            "cost_us": float(cost),
            "clamped": int(sum(s == "clamped" for s in statuses)),
            **vc[m],
        }
    return {"held_out": int(len(held)), "train": int(len(keep)), "methods": out}


def select_method(summary: dict[str, dict]) -> tuple[str, str]:
    ok = {m: v for m, v in summary.items() if v["itd_err_us"] < ITD_LIMIT_US and v["cost_us"] < COST_LIMIT_US
          and v["vertex_nmse_db_p95"] < VERTEX_LIMIT_DB}
    pool = ok or summary
    best = min(pool, key=lambda m: pool[m]["lsd_db"])
    rejected = {m: [c for c, bad in (("ITD", v["itd_err_us"] >= ITD_LIMIT_US),
                                     ("vertex consistency", v["vertex_nmse_db_p95"] >= VERTEX_LIMIT_DB),
                                     ("cost", v["cost_us"] >= COST_LIMIT_US)) if bad]
                for m, v in summary.items() if m not in ok}
    reason = (f"lowest mean LSD ({pool[best]['lsd_db']:.2f} dB) among methods with mean ITD error < "
              f"{ITD_LIMIT_US:.0f} us, vertex-consistency NMSE (p95) < {VERTEX_LIMIT_DB:.0f} dB and "
              f"query cost < {COST_LIMIT_US:.0f} us. Rejected: "
              + "; ".join(f"{m} ({', '.join(c)})" for m, c in rejected.items()))
    if not ok:
        reason = "no method met the ITD/cost limits; " + reason
    return best, reason


def run(subjects: list[str] | None = None, fractions=(0.1, 0.25), seed: int = 7, dataset_id: str | None = None) -> dict:
    settings = get_settings()
    cfg = settings.dataset(dataset_id)
    subjects = subjects or cfg.available_subjects()
    if not subjects:
        raise RuntimeError("No subjects downloaded")
    results = {"dataset": cfg.id, "subjects": subjects, "fractions": list(fractions), "seed": seed, "runs": []}
    for sid in subjects:
        ds = get_dataset(cfg.id, sid, settings)
        pf = prepare_filters(ds, settings.rendering)
        for frac in fractions:
            res = evaluate_subject(pf, frac, seed, settings)
            res.update({"subject": sid, "fraction": frac})
            results["runs"].append(res)
            print(f"subject {sid} holdout {frac:.0%} ({res['held_out']} dirs):")
            for m, v in res["methods"].items():
                print(f"  {m:22s} LSD {v['lsd_db']:5.2f} dB  NMSE {v['nmse_db']:6.1f} dB  "
                      f"ITD err {v['itd_err_us']:5.1f} us  TOA err {v['toa_err_us']:5.1f} us  cost {v['cost_us']:6.0f} us")
    summary = {}
    for frac in fractions:
        runs = [r for r in results["runs"] if r["fraction"] == frac]
        summary[str(frac)] = {m: {k: float(np.mean([r["methods"][m][k] for r in runs]))
                                  for k in runs[0]["methods"][m]} for m in runs[0]["methods"]}
    results["summary"] = summary
    primary = str(fractions[0])
    best, reason = select_method(summary[primary])
    results["selected"] = {"method": best, "reason": reason, "based_on_fraction": fractions[0],
                           "itd_limit_us": ITD_LIMIT_US, "cost_limit_us": COST_LIMIT_US,
                           "vertex_limit_db": VERTEX_LIMIT_DB,
                           "configured_method": settings.rendering.interpolation}
    results["generated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    results["machine"] = {"python": platform.python_version(), "platform": platform.platform(),
                          "processor": platform.processor()}
    return results


def results_path() -> Path:
    return get_settings().results_dir / "interpolation_eval.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--subjects", nargs="*")
    ap.add_argument("--fractions", nargs="*", type=float, default=[0.1, 0.25])
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    res = run(args.subjects, tuple(args.fractions), args.seed, args.dataset)
    path = results_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=2))
    print(f"\nSelected: {res['selected']['method']} - {res['selected']['reason']}")
    if res["selected"]["method"] != res["selected"]["configured_method"]:
        print(f"NOTE: configured rendering.interpolation is '{res['selected']['configured_method']}'")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
