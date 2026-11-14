"""HRIR preprocessing: onset (time-of-arrival) estimation, fractional alignment,
minimum-phase decomposition, and preparation of the filters used for rendering.

Time alignment
--------------
Every measured HRIR h is modelled as an onset delay tau (TOA, in samples) followed
by a "shape" a[n] whose onset sits at a fixed ``margin``:

    h[n] = a[n - (tau - margin)]          (fractional shift, band-limited)

Interpolating the aligned shapes and the TOAs separately avoids the comb filtering
that results from averaging HRIRs whose onsets do not coincide. The fractional shift
is implemented in the frequency domain, which makes it exactly invertible (apart from
the Nyquist bin, which is real-valued by construction).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import resample_poly

from .config import RenderingConfig
from .dataset_loader import HRIRDataset


def next_pow2(n: int) -> int:
    return 1 << max(0, int(np.ceil(np.log2(max(1, n)))))


def filter_length_for(n_taps: int, block_size: int) -> int:
    """Rendering filter length: room for the HRIR plus 32 samples of slack, rounded up to
    a whole number of processing blocks (required by the partitioned convolver)."""
    need = n_taps + 32
    return int(np.ceil(need / block_size) * block_size)


def frac_shift(h: np.ndarray, shift: np.ndarray | float, n_fft: int) -> np.ndarray:
    """Delay ``h`` (..., N) by ``shift`` samples (may be fractional / negative) with a
    circular, band-limited shift on an ``n_fft``-point grid. Returns (..., n_fft).

    The Nyquist bin of a real signal must stay real, so it is scaled by cos(pi*shift)
    (the real part of the ideal phase factor). Must match frontend/src/dsp/fracshift.ts.
    """
    if n_fft % 2:
        raise ValueError("n_fft must be even")
    h = np.asarray(h, dtype=float)
    shift = np.asarray(shift, dtype=float)[..., None]
    spec = np.fft.rfft(h, n=n_fft, axis=-1)
    k = np.arange(n_fft // 2 + 1)
    phase = np.exp(-2j * np.pi * k * shift / n_fft)
    phase[..., -1] = np.cos(np.pi * shift[..., 0])
    return np.fft.irfft(spec * phase, n=n_fft, axis=-1)


def estimate_toa(h: np.ndarray, threshold_db: float = -20.0, upsample: int = 8) -> np.ndarray:
    """Onset (time of arrival) in samples of each IR along the last axis.

    The IR is band-limited upsampled by ``upsample`` and the first crossing of
    ``threshold_db`` relative to the IR's absolute peak is located with linear
    interpolation between the two neighbouring upsampled samples.
    """
    h = np.asarray(h, dtype=float)
    shape = h.shape[:-1]
    flat = h.reshape(-1, h.shape[-1])
    up = resample_poly(flat, upsample, 1, axis=-1) if upsample > 1 else flat
    mag = np.abs(up)
    peak = mag.max(axis=-1, keepdims=True)
    peak = np.where(peak <= 0, 1.0, peak)
    thr = peak * 10.0 ** (threshold_db / 20.0)
    above = mag >= thr
    idx = np.argmax(above, axis=-1)
    rows = np.arange(flat.shape[0])
    prev = np.clip(idx - 1, 0, None)
    m0 = mag[rows, prev]
    m1 = mag[rows, idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where((idx > 0) & (m1 > m0), (thr[:, 0] - m0) / (m1 - m0), 0.0)
    t_up = np.where(idx > 0, prev + np.clip(frac, 0.0, 1.0), 0.0)
    return (t_up / upsample).reshape(shape)


def itd_cross_correlation(hl: np.ndarray, hr: np.ndarray, fs: float, upsample: int = 8,
                          max_itd_s: float = 1e-3) -> float:
    """Interaural time difference (seconds, positive = right ear lags, i.e. source on the
    left) from the maximum of the interaural cross-correlation. Used to validate TOAs."""
    l = resample_poly(hl, upsample, 1)
    r = resample_poly(hr, upsample, 1)
    xc = np.correlate(r, l, mode="full")
    lags = np.arange(-len(l) + 1, len(r))
    lim = int(max_itd_s * fs * upsample)
    sel = np.abs(lags) <= lim
    return float(lags[sel][np.argmax(xc[sel])] / (fs * upsample))


def minimum_phase(h: np.ndarray, n_fft: int = 1024) -> np.ndarray:
    """Minimum-phase version of each IR (same magnitude response) via the real cepstrum."""
    h = np.asarray(h, dtype=float)
    spec = np.fft.fft(h, n=n_fft, axis=-1)
    log_mag = np.log(np.maximum(np.abs(spec), 1e-12))
    cep = np.fft.ifft(log_mag, axis=-1).real
    fold = np.zeros(n_fft)
    fold[0] = 1.0
    fold[1:n_fft // 2] = 2.0
    fold[n_fft // 2] = 1.0
    hmin = np.fft.ifft(np.exp(np.fft.fft(cep * fold, axis=-1)), axis=-1).real
    return hmin[..., : h.shape[-1]]


def align(ir: np.ndarray, toa: np.ndarray, margin: int, length: int) -> np.ndarray:
    """Shift every IR so that its onset lands on ``margin``; output length ``length``."""
    n_fft = next_pow2(2 * max(length, ir.shape[-1]))
    return frac_shift(ir, margin - toa, n_fft)[..., :length]


def reconstruct(aligned: np.ndarray, toa: np.ndarray | float, margin: int, length: int) -> np.ndarray:
    """Inverse of :func:`align`: re-insert the onset delay ``toa``."""
    n_fft = next_pow2(2 * max(length, aligned.shape[-1]))
    return frac_shift(aligned, np.asarray(toa) - margin, n_fft)[..., :length]


def fit_length(ir: np.ndarray, length: int) -> np.ndarray:
    n = ir.shape[-1]
    if n >= length:
        return ir[..., :length].copy()
    out = np.zeros(ir.shape[:-1] + (length,))
    out[..., :n] = ir
    return out


@dataclass
class PreparedFilters:
    """Everything the renderers need, at one sampling rate."""
    fs: float
    filter_length: int
    margin: int
    measured: np.ndarray      # (M, 2, L) measured HRIRs, zero padded / truncated to L
    aligned: np.ndarray       # (M, 2, L) onset-aligned shapes
    toa: np.ndarray           # (M, 2) onset in samples
    minphase: np.ndarray      # (M, 2, L) minimum-phase shapes (evaluation alternative)
    vectors: np.ndarray       # (M, 3)
    gain: float               # dataset normalisation gain
    truncated_energy_db: float  # worst-case energy discarded by fitting to L (dB re total)
    source_fs: float
    source_length: int


def resample_hrirs(ir: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if abs(fs_in - fs_out) < 1e-6:
        return ir.copy()
    from math import gcd
    a, b = int(round(fs_out)), int(round(fs_in))
    g = gcd(a, b)
    return resample_poly(ir, a // g, b // g, axis=-1)


def prepare_filters(ds: HRIRDataset, cfg: RenderingConfig, fs: float | None = None) -> PreparedFilters:
    fs = float(fs or ds.fs)
    ir = resample_hrirs(ds.ir, ds.fs, fs)
    L = filter_length_for(ir.shape[-1], cfg.block_size)
    total = np.sum(ir ** 2, axis=-1)
    kept = np.sum(fit_length(ir, L) ** 2, axis=-1)
    with np.errstate(divide="ignore"):
        lost = 10 * np.log10(np.maximum((total - kept) / np.maximum(total, 1e-30), 1e-30))
    toa = estimate_toa(ir, cfg.onset_threshold_db, cfg.onset_upsample)
    aligned = align(ir, toa, cfg.align_margin, L)
    measured = fit_length(ir, L)
    mp = minimum_phase(ir, n_fft=next_pow2(4 * L))
    minph = fit_length(mp, L)
    # Normalisation: the loudest ear filter gets unit energy, so white noise at the input
    # never comes out louder (RMS) than it went in.
    gain = float(1.0 / np.sqrt(np.max(np.sum(measured ** 2, axis=-1))))
    return PreparedFilters(fs=fs, filter_length=L, margin=cfg.align_margin, measured=measured,
                           aligned=aligned, toa=toa, minphase=minph, vectors=ds.vectors.copy(),
                           gain=gain, truncated_energy_db=float(np.max(lost)), source_fs=ds.fs,
                           source_length=ds.hrir_length)
