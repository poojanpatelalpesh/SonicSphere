"""Binaural renderers.

* :func:`render_static`  - reference: y_L = x * h_L, y_R = x * h_R for one direction
                           (full linear convolution, direct or FFT).
* :class:`BlockEngine`   - the real-time algorithm, block by block, exactly as the
                           browser AudioWorklet runs it: per-block direction lookup on a
                           quantised cache grid, filter interpolation, linear output
                           crossfade between the previous and the new filter whenever
                           the filter changes, per-block gain ramps and a peak limiter.
                           Used to render moving sources and to produce golden vectors
                           for the TypeScript engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .audio_processing import PeakLimiter
from .convolution import DirectBlockConvolver, PartitionedConvolver, binaural_convolve
from .coordinate_system import cart_to_sph, quantize_direction, rotate_about_up, world_to_head
from .hrir_interpolation import HRIRInterpolator, InterpolatedHRIR
from .trajectory import Trajectory

MIN_DISTANCE_M = 0.25
DIST_GAIN_MIN, DIST_GAIN_MAX = 0.25, 2.0


def distance_gain(r: float, enabled: bool = True, ref_m: float = 1.0) -> float:
    """1/r law relative to the HRIR measurement distance, clamped. HRIRs are far-field;
    near-field (< ~1 m) spectral effects are not modelled."""
    if not enabled:
        return 1.0
    return float(min(DIST_GAIN_MAX, max(DIST_GAIN_MIN, ref_m / max(r, MIN_DISTANCE_M))))


def render_static(x: np.ndarray, hrir: np.ndarray, method: str = "direct", gain: float = 1.0) -> np.ndarray:
    """Reference binaural render, output length len(x) + L - 1, columns (left, right)."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("render_static expects a mono signal")
    if x.size == 0:
        return np.zeros((0, 2))
    return gain * binaural_convolve(x, hrir, method=method)


@dataclass
class VoiceState:
    conv: DirectBlockConvolver | PartitionedConvolver
    key: tuple[float, float] | None = None
    filt: np.ndarray | None = None          # (2, L) time-domain filter (direct)
    spectra: np.ndarray | None = None       # (2, P, B+1) partition spectra (partitioned)
    info: InterpolatedHRIR | None = None
    gain: float | None = None


@dataclass
class EngineStats:
    blocks: int = 0
    filter_updates: int = 0
    crossfades: int = 0
    keys: list = field(default_factory=list)


class BlockEngine:
    def __init__(self, interp: HRIRInterpolator, block_size: int = 128, method: str = "direct",
                 crossfade: bool = True, limiter_threshold: float = 0.98, limiter_release_ms: float = 80.0,
                 quantum_deg: float = 0.1, n_voices: int = 1, record_keys: bool = False):
        if method not in ("direct", "partitioned"):
            raise ValueError("method must be 'direct' or 'partitioned'")
        self.interp = interp
        self.pf = interp.pf
        self.B = block_size
        self.L = self.pf.filter_length
        self.method = method
        self.crossfade = crossfade
        self.quantum = quantum_deg
        self.limiter = PeakLimiter(self.pf.fs, limiter_threshold, limiter_release_ms)
        mk = (lambda: DirectBlockConvolver(self.L, self.B)) if method == "direct" else \
            (lambda: PartitionedConvolver(self.L, self.B))
        self.voices = [VoiceState(mk()) for _ in range(n_voices)]
        self.ramp = (np.arange(1, self.B + 1) / self.B)
        self.stats = EngineStats()
        self.record_keys = record_keys

    def _filter_for(self, v: VoiceState, key: tuple[float, float]):
        # The filter is always computed for the quantised direction (cached), exactly like
        # the browser, so both implementations produce identical filters.
        info = self.interp._cached(key[0], key[1])
        filt = info.hrir.astype(float)
        spectra = None
        if self.method == "partitioned":
            spectra = np.stack([v.conv.partition(filt[0]), v.conv.partition(filt[1])])
        return info, filt, spectra

    def _apply(self, v: VoiceState, filt, spectra) -> np.ndarray:
        if self.method == "direct":
            return np.stack([v.conv.apply(filt[0]), v.conv.apply(filt[1])])
        return np.stack([v.conv.apply(spectra[0]), v.conv.apply(spectra[1])])

    def process_block(self, blocks: list[np.ndarray], directions: list[np.ndarray], gains: list[float]) -> np.ndarray:
        """One block. ``blocks[i]`` is voice i's input (B,), ``directions[i]`` its head-frame
        direction, ``gains[i]`` its target linear gain. Returns (B, 2)."""
        out = np.zeros((2, self.B))
        for v, xb, d, g in zip(self.voices, blocks, directions, gains):
            v.conv.push(np.asarray(xb, dtype=float))
            az, el, _ = cart_to_sph(np.asarray(d, dtype=float))
            key = quantize_direction(float(az), float(el), self.quantum)
            if self.record_keys:
                self.stats.keys.append(key)
            if key != v.key:
                info, filt, spectra = self._filter_for(v, key)
                y_new = self._apply(v, filt, spectra)
                if v.key is not None and self.crossfade:
                    y_old = self._apply(v, v.filt, v.spectra)
                    y = y_old * (1.0 - self.ramp) + y_new * self.ramp
                    self.stats.crossfades += 1
                else:
                    y = y_new
                v.key, v.filt, v.spectra, v.info = key, filt, spectra, info
                self.stats.filter_updates += 1
            else:
                y = self._apply(v, v.filt, v.spectra)
            g0 = g if v.gain is None else v.gain
            out += y * (g0 + (g - g0) * self.ramp)
            v.gain = g
        self.stats.blocks += 1
        return self.limiter.process(out.T)


@dataclass
class ListenerPose:
    position: np.ndarray
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0


def render_moving(signals: list[np.ndarray], azimuth_offsets_deg: list[float], interp: HRIRInterpolator,
                  source_position: Callable[[int], np.ndarray], pose: ListenerPose, gain: float = 1.0,
                  block_size: int = 128, method: str = "direct", crossfade: bool = True,
                  distance_attenuation: bool = True, tail: bool = True, record_keys: bool = False,
                  limiter_threshold: float = 0.98, limiter_release_ms: float = 80.0) -> tuple[np.ndarray, BlockEngine]:
    """Render a (possibly moving) source with :class:`BlockEngine`.

    ``source_position(frame)`` returns the world-space source position at the first
    sample of each block. Output has len + L - 1 samples when ``tail`` is set.
    """
    n = max(s.size for s in signals)
    eng = BlockEngine(interp, block_size, method, crossfade, limiter_threshold, limiter_release_ms,
                      interp.quantum, n_voices=len(signals), record_keys=record_keys)
    total = n + (eng.L - 1 if tail else 0)
    n_blocks = (total + block_size - 1) // block_size
    padded = []
    for s in signals:
        p = np.zeros(n_blocks * block_size)
        p[: s.size] = s
        padded.append(p)
    out = np.zeros((n_blocks * block_size, 2))
    for b in range(n_blocks):
        frame = b * block_size
        d = world_to_head(source_position(frame), pose.position, pose.yaw_deg, pose.pitch_deg)
        r = float(np.linalg.norm(d))
        d = d / r if r > 1e-9 else np.array([1.0, 0.0, 0.0])
        g = gain * distance_gain(r, distance_attenuation)
        dirs = [rotate_about_up(d, off) if off else d for off in azimuth_offsets_deg]
        out[frame: frame + block_size] = eng.process_block(
            [p[frame: frame + block_size] for p in padded], dirs, [g] * len(signals))
    return out[:total], eng


def render_trajectory(signals: list[np.ndarray], azimuth_offsets_deg: list[float], interp: HRIRInterpolator,
                      traj: Trajectory, pose: ListenerPose, **kw) -> tuple[np.ndarray, BlockEngine]:
    fs = interp.pf.fs
    return render_moving(signals, azimuth_offsets_deg, interp, lambda f: traj.position_at(f / fs), pose, **kw)
