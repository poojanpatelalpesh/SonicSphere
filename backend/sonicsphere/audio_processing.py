"""Audio input handling: decoding, validation, resampling, stereo policy,
normalisation and output peak limiting.

Stereo input policy (explicit, never silent)
--------------------------------------------
A binaural renderer spatialises point sources, so a stereo file must be mapped to one
or more mono source signals. The policy is always reported back to the user:

* ``mid``         (default) s = (L + R) / 2. Keeps everything that is common to both
                  channels. The analysis reports how much energy the side signal
                  (L - R) / 2 carried, i.e. what this policy removes; if the channels
                  are strongly anti-correlated the user is warned.
* ``left`` / ``right``  use one channel.
* ``stereo_pair`` render L and R as two virtual sources at +/- width/2 around the
                  chosen direction (L to the listener's left). Nothing is discarded.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

STEREO_POLICIES = ("mid", "left", "right", "stereo_pair")


class AudioInputError(ValueError):
    pass


@dataclass
class SourceSignals:
    signals: list[np.ndarray]           # one mono signal per virtual source
    azimuth_offsets_deg: list[float]    # offset of each virtual source from the source direction
    fs: float
    policy: str
    analysis: dict = field(default_factory=dict)


def decode_audio(data: bytes, max_bytes: int, max_duration_s: float, min_duration_s: float
                 ) -> tuple[np.ndarray, float]:
    """Decode WAV/FLAC/OGG/MP3 (whatever libsndfile supports) into (frames, channels)."""
    if len(data) == 0:
        raise AudioInputError("The uploaded file is empty")
    if len(data) > max_bytes:
        raise AudioInputError(f"File is {len(data) / 1e6:.1f} MB; the limit is {max_bytes / 1e6:.1f} MB")
    try:
        x, fs = sf.read(io.BytesIO(data), dtype="float64", always_2d=True)
    except Exception as exc:
        raise AudioInputError(f"Could not decode audio ({exc}). Supported: WAV, FLAC, OGG, MP3.") from exc
    if x.shape[0] == 0:
        raise AudioInputError("The audio file contains no samples")
    dur = x.shape[0] / fs
    if dur > max_duration_s:
        raise AudioInputError(f"Audio is {dur:.1f} s long; the limit is {max_duration_s:.0f} s")
    if dur < min_duration_s:
        raise AudioInputError(f"Audio is only {dur * 1000:.1f} ms long; the minimum is {min_duration_s * 1000:.0f} ms")
    if x.shape[1] > 2:
        raise AudioInputError(f"{x.shape[1]}-channel audio is not supported; use mono or stereo")
    if not np.all(np.isfinite(x)):
        raise AudioInputError("Audio contains NaN/Inf samples")
    return x, float(fs)


def resample(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Polyphase resampling (anti-aliased) along axis 0."""
    if abs(fs_in - fs_out) < 1e-6:
        return np.asarray(x, dtype=float)
    a, b = int(round(fs_out)), int(round(fs_in))
    g = gcd(a, b)
    return resample_poly(np.asarray(x, dtype=float), a // g, b // g, axis=0)


def analyse_stereo(x: np.ndarray) -> dict:
    """Inter-channel statistics that justify / warn about the stereo policy."""
    if x.shape[1] == 1:
        return {"channels": 1}
    l, r = x[:, 0], x[:, 1]
    mid, side = 0.5 * (l + r), 0.5 * (l - r)
    e_mid, e_side = float(np.sum(mid ** 2)), float(np.sum(side ** 2))
    tot = e_mid + e_side
    denom = np.sqrt(np.sum(l ** 2) * np.sum(r ** 2))
    corr = float(np.sum(l * r) / denom) if denom > 0 else 1.0
    return {
        "channels": 2,
        "correlation": corr,
        "side_energy_fraction": e_side / tot if tot > 0 else 0.0,
        "identical_channels": bool(np.allclose(l, r)),
        "warning": ("Channels are strongly anti-correlated; a mid downmix would cancel. "
                    "Use 'stereo_pair', 'left' or 'right'.") if corr < -0.5 else None,
    }


def apply_stereo_policy(x: np.ndarray, fs: float, policy: str = "mid", width_deg: float = 30.0) -> SourceSignals:
    if policy not in STEREO_POLICIES:
        raise AudioInputError(f"Unknown stereo policy '{policy}'. Choose from {STEREO_POLICIES}")
    analysis = analyse_stereo(x)
    if x.shape[1] == 1:
        return SourceSignals([x[:, 0].copy()], [0.0], fs, "mono", analysis)
    if policy == "mid":
        return SourceSignals([0.5 * (x[:, 0] + x[:, 1])], [0.0], fs, policy, analysis)
    if policy == "left":
        return SourceSignals([x[:, 0].copy()], [0.0], fs, policy, analysis)
    if policy == "right":
        return SourceSignals([x[:, 1].copy()], [0.0], fs, policy, analysis)
    half = width_deg / 2.0  # positive azimuth = left
    return SourceSignals([x[:, 0].copy(), x[:, 1].copy()], [half, -half], fs, policy, analysis)


def normalize_sources(signals: list[np.ndarray], target_rms_db: float = -20.0, peak_ceiling_db: float = -1.0
                      ) -> tuple[list[np.ndarray], float]:
    """One common gain for all virtual sources: reach the RMS target unless that would
    push the peak above the ceiling. Silence is returned unchanged (gain 1)."""
    allx = np.concatenate([s.ravel() for s in signals]) if signals else np.zeros(0)
    if allx.size == 0:
        return signals, 1.0
    rms = float(np.sqrt(np.mean(allx ** 2)))
    peak = float(np.max(np.abs(allx)))
    if peak <= 1e-9:
        return signals, 1.0
    g = 10 ** (target_rms_db / 20) / max(rms, 1e-12)
    g = min(g, 10 ** (peak_ceiling_db / 20) / peak)
    return [s * g for s in signals], g


def ensure_min_length(x: np.ndarray, n_min: int) -> np.ndarray:
    if x.shape[0] >= n_min:
        return x
    pad = [(0, n_min - x.shape[0])] + [(0, 0)] * (x.ndim - 1)
    return np.pad(x, pad)


class PeakLimiter:
    """Instant-attack, exponential-release peak limiter. |output| <= threshold always.
    Mirrors frontend/src/dsp/limiter.ts."""

    def __init__(self, fs: float, threshold: float = 0.98, release_ms: float = 80.0):
        self.threshold = threshold
        self.coef = float(np.exp(-1.0 / (release_ms * 1e-3 * fs)))
        self.g = 1.0
        self.min_gain = 1.0

    def process(self, stereo: np.ndarray) -> np.ndarray:
        g, thr, c = self.g, self.threshold, self.coef
        peaks = np.max(np.abs(stereo), axis=1)
        if g == 1.0 and (peaks.size == 0 or peaks.max() <= thr):
            return stereo.copy()          # fast path: limiter idle (identical result)
        out = np.empty_like(stereo)
        for n in range(stereo.shape[0]):
            pk = peaks[n]
            target = thr / pk if pk > thr else 1.0
            g = target if target < g else target + (g - target) * c
            out[n] = stereo[n] * g
            if g < self.min_gain:
                self.min_gain = g
        self.g = g
        return out


def to_wav_bytes(stereo: np.ndarray, fs: float, subtype: str = "PCM_16") -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(stereo, -1.0, 1.0), int(round(fs)), format="WAV", subtype=subtype)
    return buf.getvalue()
