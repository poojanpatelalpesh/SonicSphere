"""Dataset adapter for SOFA (AES69) HRIR/HRTF files.

CIPIC is distributed by the SOFA conventions project as ``SimpleFreeFieldHRIR`` files
(netCDF-4 / HDF5). This module reads them with :mod:`h5py`, validates the structure and
returns an :class:`HRIRDataset` expressed in the SOFA head frame
(x = front, y = left, z = up) with the left ear in channel 0.

Frequency-domain ``SimpleFreeFieldHRTF`` files are also accepted; they are converted
to time-domain HRIRs with an inverse real FFT (see :func:`hrtf_to_hrir`).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .config import DatasetConfig, Settings, get_settings
from .coordinate_system import cart_to_interaural_polar, cart_to_sph, sph_to_cart

HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


class DatasetError(RuntimeError):
    """Raised for missing, unreadable or malformed dataset files."""


@dataclass
class HRIRDataset:
    dataset_id: str
    subject: str
    source_file: Path
    fs: float
    ir: np.ndarray            # (M, 2, N) float64, channel 0 = left ear, 1 = right ear
    vectors: np.ndarray       # (M, 3) unit direction vectors in the head frame
    azimuth: np.ndarray       # (M,) SOFA azimuth, degrees [0, 360)
    elevation: np.ndarray     # (M,) SOFA elevation, degrees
    distance: np.ndarray      # (M,) metres
    conventions: str
    attributes: dict[str, str] = field(default_factory=dict)
    sha256: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def num_directions(self) -> int:
        return int(self.ir.shape[0])

    @property
    def hrir_length(self) -> int:
        return int(self.ir.shape[2])

    def interaural_polar(self) -> tuple[np.ndarray, np.ndarray]:
        return cart_to_interaural_polar(self.vectors)


def _attr(obj: Any, name: str, default: str = "") -> str:
    if name not in obj.attrs:
        return default
    v = obj.attrs[name]
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.ndarray) and v.dtype.kind in "SO":
        return " ".join(x.decode() if isinstance(x, bytes) else str(x) for x in v.ravel())
    try:
        return str(v)
    except Exception:  # pragma: no cover - exotic attribute types
        return default


def _require(f: h5py.File, name: str) -> np.ndarray:
    if name not in f:
        raise DatasetError(f"SOFA file is missing required variable '{name}'")
    return np.asarray(f[name][()])


def _positions_to_vectors(pos: np.ndarray, ptype: str, units: str) -> np.ndarray:
    pos = np.asarray(pos, dtype=float)
    if pos.ndim == 3:  # some writers keep a trailing singleton dimension
        pos = pos[..., 0]
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise DatasetError(f"SourcePosition must have shape (M, 3); got {pos.shape}")
    ptype = ptype.lower()
    if ptype == "spherical":
        az, el = pos[:, 0], pos[:, 1]
        if "rad" in units.lower() and "degree" not in units.lower():
            az, el = np.rad2deg(az), np.rad2deg(el)
        r = pos[:, 2]
        return sph_to_cart(az, el, np.where(r > 0, r, 1.0))
    if ptype == "cartesian":
        return pos
    raise DatasetError(f"Unsupported SourcePosition Type '{ptype}'")


def _head_rotation(view: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Rotation whose rows are the head x/y/z axes in SOFA global coordinates."""
    x = view / np.linalg.norm(view)
    z = up - np.dot(up, x) * x
    if np.linalg.norm(z) < 1e-9:
        raise DatasetError("ListenerView and ListenerUp are parallel")
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def hrtf_to_hrir(real: np.ndarray, imag: np.ndarray, freqs: np.ndarray, fs: float | None = None
                 ) -> tuple[np.ndarray, float, list[str]]:
    """Reconstruct real time-domain HRIRs from one-sided complex spectra.

    The spectra must be sampled on the uniform DFT grid 0, df, ..., K*df. If the last
    bin is the Nyquist frequency the IR length is 2K, otherwise 2K+1 (odd length).
    DC (and Nyquist) bins of a real signal must be real; any imaginary residue there
    is reported and discarded, which is exactly what conjugate symmetry requires.
    """
    notes: list[str] = []
    real = np.asarray(real, dtype=float)
    imag = np.asarray(imag, dtype=float)
    freqs = np.asarray(freqs, dtype=float).ravel()
    k = freqs.size
    if real.shape != imag.shape or real.shape[-1] != k:
        raise DatasetError("HRTF real/imag/frequency dimensions do not agree")
    if k < 2 or abs(freqs[0]) > 1e-6:
        raise DatasetError("HRTF frequency vector must start at 0 Hz")
    df = np.diff(freqs)
    if not np.allclose(df, df[0], rtol=1e-6):
        raise DatasetError("HRTF frequencies are not uniformly spaced; cannot use an inverse DFT")
    spec = real + 1j * imag
    if fs is None or fs <= 0:
        fs = 2.0 * freqs[-1]          # assume the last bin is Nyquist
    n_even = 2 * (k - 1)
    if np.isclose(freqs[-1], fs / 2.0, rtol=1e-6):
        n = n_even
        if np.max(np.abs(spec[..., -1].imag)) > 1e-9:
            notes.append("Nyquist bin had a non-zero imaginary part; discarded (conjugate symmetry).")
    else:
        n = n_even + 1
    if not np.isclose(df[0], fs / n, rtol=1e-4):
        raise DatasetError(f"Frequency spacing {df[0]:.3f} Hz is inconsistent with fs={fs} and N={n}")
    if np.max(np.abs(spec[..., 0].imag)) > 1e-9:
        notes.append("DC bin had a non-zero imaginary part; discarded (conjugate symmetry).")
    ir = np.fft.irfft(spec, n=n, axis=-1)
    return ir, float(fs), notes


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sofa(path: str | Path, dataset_id: str = "sofa", subject: str = "") -> HRIRDataset:
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"Dataset file not found: {path}")
    with path.open("rb") as fh:
        if fh.read(8) != HDF5_MAGIC:
            raise DatasetError(f"{path.name} is not an HDF5/netCDF-4 SOFA file")
    try:
        f = h5py.File(path, "r")
    except OSError as exc:
        raise DatasetError(f"Cannot open {path.name}: {exc}") from exc

    notes: list[str] = []
    with f:
        conventions = _attr(f, "SOFAConventions")
        if _attr(f, "Conventions") != "SOFA":
            raise DatasetError(f"{path.name}: global attribute Conventions is not 'SOFA'")
        attributes = {k: _attr(f, k) for k in f.attrs.keys() if not k.startswith("_")}

        if conventions == "SimpleFreeFieldHRIR" or ("Data.IR" in f and conventions != "SimpleFreeFieldHRTF"):
            ir = np.asarray(_require(f, "Data.IR"), dtype=float)
            fs_arr = np.atleast_1d(_require(f, "Data.SamplingRate")).astype(float)
            if fs_arr.size != 1:
                raise DatasetError("Only a single global sampling rate is supported")
            fs = float(fs_arr[0])
            units = _attr(f["Data.SamplingRate"], "Units", "hertz").lower()
            if units not in ("hertz", "hz", ""):
                raise DatasetError(f"Unsupported sampling-rate unit '{units}'")
        elif conventions == "SimpleFreeFieldHRTF":
            real = _require(f, "Data.Real")
            imag = _require(f, "Data.Imag")
            freqs = _require(f, "N")
            ir, fs, hn = hrtf_to_hrir(real, imag, freqs)
            notes.extend(hn)
            notes.append("Time-domain HRIRs reconstructed from HRTFs by inverse real FFT.")
        else:
            raise DatasetError(f"Unsupported SOFA convention '{conventions}'")

        if ir.ndim != 3:
            raise DatasetError(f"IR data must be 3-D (M, R, N); got shape {ir.shape}")
        m, r, n = ir.shape
        if r != 2:
            raise DatasetError(f"Expected 2 receivers (ears); got {r}")
        if n < 8:
            raise DatasetError(f"HRIR length {n} is implausibly short")
        if not np.all(np.isfinite(ir)):
            raise DatasetError("IR data contains NaN or Inf values")
        if not (8000.0 <= fs <= 384000.0):
            raise DatasetError(f"Implausible sampling rate {fs}")

        src = f["SourcePosition"] if "SourcePosition" in f else None
        if src is None:
            raise DatasetError("SOFA file is missing 'SourcePosition'")
        vectors = _positions_to_vectors(src[()], _attr(src, "Type", "spherical"),
                                        _attr(src, "Units", "degree, degree, metre"))
        if vectors.shape[0] == 1 and m > 1:
            raise DatasetError("A single SourcePosition for many measurements is not a free-field HRIR set")
        if vectors.shape[0] != m:
            raise DatasetError(f"SourcePosition has {vectors.shape[0]} rows but IR data has {m}")

        # Express positions relative to the listener's head orientation.
        if "ListenerView" in f:
            view = np.asarray(f["ListenerView"][()], dtype=float).reshape(-1, 3)[0]
            vtype = _attr(f["ListenerView"], "Type", "cartesian").lower()
            if vtype == "spherical":
                view = sph_to_cart(view[0], view[1], 1.0)
            up = (np.asarray(f["ListenerUp"][()], dtype=float).reshape(-1, 3)[0]
                  if "ListenerUp" in f else np.array([0.0, 0.0, 1.0]))
            rot = _head_rotation(view, up)
            if not np.allclose(rot, np.eye(3), atol=1e-9):
                notes.append("Source positions rotated into the listener's head frame (non-default ListenerView/Up).")
            if "ListenerPosition" in f:
                lp = np.asarray(f["ListenerPosition"][()], dtype=float).reshape(-1, 3)
                if lp.shape[0] == 1 and np.linalg.norm(lp[0]) > 1e-9 and _attr(src, "Type").lower() == "cartesian":
                    vectors = vectors - lp[0]
            vectors = vectors @ rot.T

        # Ear order: determine from ReceiverPosition (+y = left) rather than assuming.
        if "ReceiverPosition" in f:
            rp = np.asarray(f["ReceiverPosition"][()], dtype=float).reshape(2, -1)[:, :3]
            if rp.shape[1] == 3 and abs(rp[0, 1] - rp[1, 1]) > 1e-6 and rp[0, 1] < rp[1, 1]:
                ir = ir[:, ::-1, :]
                notes.append("Receivers reordered so that channel 0 is the left ear.")

        # Broadband delays (SOFA Data.Delay, in samples) are part of the response.
        if "Data.Delay" in f:
            delay = np.asarray(f["Data.Delay"][()], dtype=float)
            if np.any(np.abs(delay) > 0):
                delay = np.broadcast_to(delay.reshape(-1, 2), (m, 2))
                int_delay = np.round(delay).astype(int)
                if np.any(int_delay < 0):
                    raise DatasetError("Negative Data.Delay values are not supported")
                pad = int(int_delay.max())
                padded = np.zeros((m, 2, n + pad))
                for i in range(m):
                    for e in range(2):
                        padded[i, e, int_delay[i, e]: int_delay[i, e] + n] = ir[i, e]
                ir = padded
                notes.append(f"Applied Data.Delay (up to {pad} samples, rounded to integers).")

        radius = np.linalg.norm(vectors, axis=1)
        if np.any(radius <= 0):
            raise DatasetError("SourcePosition contains a zero-length direction")
        unit = vectors / radius[:, None]
        az, el, _ = cart_to_sph(unit)

    return HRIRDataset(
        dataset_id=dataset_id,
        subject=subject,
        source_file=path,
        fs=fs,
        ir=np.ascontiguousarray(ir),
        vectors=unit,
        azimuth=az,
        elevation=el,
        distance=radius,
        conventions=conventions or "SimpleFreeFieldHRIR",
        attributes=attributes,
        sha256=file_sha256(path),
        notes=notes,
    )


def load_dataset(cfg: DatasetConfig, subject: str | None = None) -> HRIRDataset:
    sid = subject or cfg.default_subject
    if cfg.format != "sofa":
        raise DatasetError(f"Dataset '{cfg.id}' uses unsupported format '{cfg.format}'")
    path = cfg.subject_path(sid)
    if not path.is_file():
        raise DatasetError(
            f"Subject '{sid}' of dataset '{cfg.id}' is not downloaded (expected {path}). "
            f"Run: python -m sonicsphere.download_dataset --dataset {cfg.id} --subjects {sid}")
    return load_sofa(path, dataset_id=cfg.id, subject=sid)


@lru_cache(maxsize=8)
def _cached(dataset_id: str, subject: str, mtime: float) -> HRIRDataset:
    settings = get_settings()
    return load_dataset(settings.dataset(dataset_id), subject)


def get_dataset(dataset_id: str | None = None, subject: str | None = None,
                settings: Settings | None = None) -> HRIRDataset:
    settings = settings or get_settings()
    cfg = settings.dataset(dataset_id)
    sid = subject or cfg.default_subject
    path = cfg.subject_path(sid)
    if not path.is_file():
        return load_dataset(cfg, sid)  # raises a helpful DatasetError
    return _cached(cfg.id, sid, path.stat().st_mtime)


def describe_grid(ds: HRIRDataset) -> dict[str, Any]:
    lat, pol = ds.interaural_polar()
    laterals = sorted({round(float(v), 3) + 0.0 for v in lat})
    return {
        "num_directions": ds.num_directions,
        "azimuth_range_deg": [float(ds.azimuth.min()), float(ds.azimuth.max())],
        "elevation_range_deg": [float(ds.elevation.min()), float(ds.elevation.max())],
        "lateral_angles_deg": laterals if len(laterals) <= 64 else None,
        "num_lateral_angles": len(laterals),
        "num_polar_angles": len({round(float(v), 3) + 0.0 for v in pol}),
        "polar_angle_range_deg": [float(pol.min()), float(pol.max())],
        "distance_m": sorted({round(float(v), 3) for v in ds.distance}),
    }
