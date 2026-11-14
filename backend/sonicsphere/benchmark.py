"""Benchmarks: direct vs FFT-based convolution, block processing cost, memory, and
the cost of the full block engine. Writes ``results/benchmark.json``.

All timings are medians of repeated runs of the *Python* implementations on this
machine. The browser's real-time path is benchmarked separately in the browser (see the
technical panel); the two are reported separately and never mixed.

Run:  python -m sonicsphere.benchmark [--quick]
"""

from __future__ import annotations

import argparse
import json
import platform
import time
import tracemalloc
from pathlib import Path

import numpy as np

from .audio_renderer import ListenerPose, render_moving
from .config import get_settings
from .convolution import (DirectBlockConvolver, PartitionedConvolver, direct_convolve, fft_convolve,
                          overlap_add, overlap_save, partitioned_convolve)
from .filter_pack import get_prepared
from .hrir_interpolation import HRIRInterpolator

METHODS = {
    "direct": direct_convolve,
    "fft": fft_convolve,
    "overlap_add": lambda x, h: overlap_add(x, h, 1024),
    "overlap_save": overlap_save,
    "partitioned_128": lambda x, h: partitioned_convolve(x, h, 128),
}


def _time(fn, repeats: int) -> float:
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def _peak_mem(fn) -> int:
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return int(peak)


def offline_benchmark(h: np.ndarray, lengths: list[int], repeats: int) -> list[dict]:
    rng = np.random.default_rng(0)
    rows = []
    for n in lengths:
        x = rng.standard_normal(n)
        ref = direct_convolve(x, h)
        scale = np.max(np.abs(ref))
        for name, fn in METHODS.items():
            if name == "partitioned_128" and n > 200_000:
                reps = 1
            else:
                reps = repeats if n <= 50_000 else max(1, repeats // 3)
            y = fn(x, h)
            err = float(np.max(np.abs(y - ref)) / scale)
            rows.append({
                "signal_length": n,
                "filter_length": int(h.size),
                "method": name,
                "time_ms": _time(lambda: fn(x, h), reps) * 1e3,
                "max_rel_error": err,
                "error_db": float(20 * np.log10(max(err, 1e-300))),
                "peak_memory_kb": _peak_mem(lambda: fn(x, h)) / 1024,
                "output_length_ok": bool(y.size == n + h.size - 1),
            })
    return rows


def block_benchmark(h: np.ndarray, fs: float, block_sizes: list[int], n_blocks: int) -> list[dict]:
    rng = np.random.default_rng(1)
    rows = []
    for B in block_sizes:
        L = int(np.ceil(h.size / B) * B)
        hp = np.zeros(L)
        hp[: h.size] = h
        xs = rng.standard_normal((n_blocks, B))
        d = DirectBlockConvolver(L, B)
        p = PartitionedConvolver(L, B)
        H = p.partition(hp)

        def run_direct():
            for xb in xs:
                d.push(xb)
                d.apply(hp)
                d.apply(hp)

        def run_part():
            for xb in xs:
                p.push(xb)
                p.apply(H)
                p.apply(H)

        deadline_us = B / fs * 1e6
        for name, fn in (("direct", run_direct), ("partitioned", run_part)):
            t = _time(fn, 3) / n_blocks * 1e6
            rows.append({"block_size": B, "filter_length": L, "method": name, "us_per_block_two_ears": t,
                         "deadline_us": deadline_us, "load_percent": 100 * t / deadline_us})
    return rows


def engine_benchmark(prep, settings, seconds: float = 2.0) -> list[dict]:
    pf = prep.filters
    interp = HRIRInterpolator(pf, settings.rendering.interpolation, settings.rendering.gap_edge_threshold_deg,
                              settings.rendering.measured_snap_deg, triangulation=prep.triangulation,
                              direction_quantum_deg=settings.rendering.direction_quantum_deg)
    rng = np.random.default_rng(2)
    n = int(seconds * pf.fs)
    x = rng.standard_normal(n) * 0.1
    pose = ListenerPose(np.zeros(3))
    rows = []
    for method in ("direct", "partitioned"):
        # A source orbiting once per second: a new filter (and a crossfade) almost every block.
        def pos(frame, fs=pf.fs):
            a = 2 * np.pi * frame / fs
            return np.array([np.sin(a), 0.3 * np.sin(3 * a), -np.cos(a)]) * 1.5
        interp._cached.cache_clear()
        t0 = time.perf_counter()
        _, eng = render_moving([x], [0.0], interp, pos, pose, gain=pf.gain, block_size=settings.rendering.block_size,
                               method=method, tail=False)
        dt = time.perf_counter() - t0
        blocks = eng.stats.blocks
        rows.append({"method": method, "blocks": blocks, "filter_updates": eng.stats.filter_updates,
                     "crossfades": eng.stats.crossfades, "us_per_block": dt / blocks * 1e6,
                     "deadline_us": settings.rendering.block_size / pf.fs * 1e6,
                     "realtime_factor": (n / pf.fs) / dt})
    return rows


def run(quick: bool = False) -> dict:
    settings = get_settings()
    prep = get_prepared()
    pf = prep.filters
    i0 = int(np.argmax(pf.vectors @ np.array([0.5, 0.8, 0.1])))
    h = pf.measured[i0, 0]
    h_raw = h[: pf.source_length]
    lengths = [128, 1024, 8192, 44100] if quick else [128, 1024, 8192, 44100, 441000]
    res = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "machine": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
                    "processor": platform.processor()},
        "dataset": prep.dataset.dataset_id, "subject": prep.dataset.subject, "fs": pf.fs,
        "note": "Python reference implementations; browser real-time timings are measured in the browser.",
        "offline": offline_benchmark(h_raw, lengths, 5 if quick else 9)
                   + offline_benchmark(h, [44100], 5 if quick else 9),
        "blocks": block_benchmark(h, pf.fs, [64, 128, 256, 512], 100 if quick else 400),
        "engine": engine_benchmark(prep, settings, 1.0 if quick else 2.0),
    }
    return res


def results_path() -> Path:
    return get_settings().results_dir / "benchmark.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    res = run(args.quick)
    print(f"Offline convolution (filter {res['offline'][0]['filter_length']} taps):")
    for r in res["offline"]:
        print(f"  N={r['signal_length']:>7} L={r['filter_length']} {r['method']:16s} {r['time_ms']:9.3f} ms  "
              f"err {r['error_db']:7.1f} dB  mem {r['peak_memory_kb']:9.1f} KiB")
    print("Block processing (two ears, two filters per block = crossfade worst case):")
    for r in res["blocks"]:
        print(f"  B={r['block_size']:4d} {r['method']:12s} {r['us_per_block_two_ears']:8.1f} us / "
              f"{r['deadline_us']:7.1f} us deadline ({r['load_percent']:.1f}%)")
    print("Full block engine (interpolation + crossfade + limiter), orbiting source:")
    for r in res["engine"]:
        print(f"  {r['method']:12s} {r['us_per_block']:8.1f} us/block, x{r['realtime_factor']:.1f} real time, "
              f"{r['filter_updates']} filter updates / {r['blocks']} blocks")
    path = results_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=2))
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
