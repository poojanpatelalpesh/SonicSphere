"""Binary "filter pack" exchanged between the backend and the browser.

Layout (little endian)::

    offset 0   4 bytes   magic  b"SSFP"
    offset 4   uint32    format version (1)
    offset 8   uint32    H = length of the JSON header in bytes (padded to a multiple of 16)
    offset 12  uint32    reserved (0)
    offset 16  H bytes   UTF-8 JSON header (space padded)
    offset 16+H          data region: arrays back to back, each 16-byte aligned

The header lists every array as ``{"dtype", "shape", "offset", "byteLength"}`` with
``offset`` relative to the start of the data region. Arrays:

    measured      float32 (M, 2, L)   measured HRIRs (left, right), zero-padded to L
    aligned       float32 (M, 2, L)   onset-aligned HRIR shapes (onset at `margin`)
    toa           float32 (M, 2)      onset (time of arrival) in samples
    vertices      float64 (M, 3)      unit direction vectors, head frame (x fwd, y left, z up)
    azel          float32 (M, 2)      SOFA azimuth / elevation of every measurement (deg)
    triangles     uint32  (T, 3)      spherical Delaunay triangles (vertex indices)
    triInverse    float64 (T, 9)      row-major inverse of [v0 v1 v2] (vertices as columns)
    triMaxEdge    float32 (T,)        longest edge of each triangle (deg)
    triGap        uint8   (T,)        1 = spans an unmeasured gap
    boundaryEdges uint32  (E, 2)      edges between measured and gap triangles

Packs are cached on disk, keyed by dataset/subject/file hash/sampling rate/config.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Settings, get_settings
from .dataset_loader import HRIRDataset, describe_grid, get_dataset
from .hrir_interpolation import Triangulation, build_triangulation
from .hrir_preprocessing import PreparedFilters, prepare_filters

MAGIC = b"SSFP"
VERSION = 1
ALIGN = 16


@dataclass
class Prepared:
    dataset: HRIRDataset
    filters: PreparedFilters
    triangulation: Triangulation


_PREPARED: dict[tuple, Prepared] = {}


def get_prepared(dataset_id: str | None = None, subject: str | None = None, fs: float | None = None,
                 settings: Settings | None = None) -> Prepared:
    settings = settings or get_settings()
    ds = get_dataset(dataset_id, subject, settings)
    fs = float(fs or ds.fs)
    key = (ds.dataset_id, ds.subject, ds.sha256, fs, settings.rendering.fingerprint())
    if key not in _PREPARED:
        pf = prepare_filters(ds, settings.rendering, fs)
        tri = build_triangulation(pf.vectors, settings.rendering.gap_edge_threshold_deg)
        _PREPARED[key] = Prepared(ds, pf, tri)
    return _PREPARED[key]


def _pad(n: int) -> int:
    return (-n) % ALIGN


def build_pack(prep: Prepared, settings: Settings) -> bytes:
    ds, pf, tri = prep.dataset, prep.filters, prep.triangulation
    r = settings.rendering
    arrays = {
        "measured": pf.measured.astype("<f4"),
        "aligned": pf.aligned.astype("<f4"),
        "toa": pf.toa.astype("<f4"),
        "vertices": pf.vectors.astype("<f8"),
        "azel": np.stack([ds.azimuth, ds.elevation], axis=1).astype("<f4"),
        "triangles": tri.triangles.astype("<u4"),
        "triInverse": tri.inverse.reshape(-1, 9).astype("<f8"),
        "triMaxEdge": tri.max_edge_deg.astype("<f4"),
        "triGap": tri.is_gap.astype("u1"),
        "boundaryEdges": tri.boundary_edges.astype("<u4"),
    }
    meta = {}
    blobs = []
    offset = 0
    for name, arr in arrays.items():
        b = np.ascontiguousarray(arr).tobytes()
        meta[name] = {"dtype": {"<f4": "float32", "<f8": "float64", "<u4": "uint32", "|u1": "uint8"}[arr.dtype.str],
                      "shape": list(arr.shape), "offset": offset, "byteLength": len(b)}
        blobs.append(b + b"\0" * _pad(len(b)))
        offset += len(b) + _pad(len(b))
    header = {
        "format": "sonicsphere-filter-pack",
        "version": VERSION,
        "dataset": ds.dataset_id,
        "datasetName": settings.dataset(ds.dataset_id).name,
        "subject": ds.subject,
        "sha256": ds.sha256,
        "fs": pf.fs,
        "sourceFs": pf.source_fs,
        "sourceLength": pf.source_length,
        "filterLength": pf.filter_length,
        "margin": pf.margin,
        "blockSize": r.block_size,
        "gain": pf.gain,
        "numDirections": int(pf.measured.shape[0]),
        "numTriangles": tri.num_triangles,
        "numGapTriangles": int(tri.is_gap.sum()),
        "gapEdgeThresholdDeg": r.gap_edge_threshold_deg,
        "measuredSnapDeg": r.measured_snap_deg,
        "directionQuantumDeg": r.direction_quantum_deg,
        "interpolation": r.interpolation,
        "limiterThreshold": r.limiter_threshold,
        "limiterReleaseMs": r.limiter_release_ms,
        "onsetThresholdDb": r.onset_threshold_db,
        "onsetUpsample": r.onset_upsample,
        "truncatedEnergyDb": pf.truncated_energy_db,
        "fracShiftFft": int(2 ** np.ceil(np.log2(2 * pf.filter_length))),
        "grid": describe_grid(ds),
        "notes": ds.notes,
        "arrays": meta,
    }
    hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hjson += b" " * _pad(len(hjson))
    return MAGIC + struct.pack("<III", VERSION, len(hjson), 0) + hjson + b"".join(blobs)


def parse_pack(data: bytes) -> tuple[dict, dict[str, np.ndarray]]:
    if data[:4] != MAGIC:
        raise ValueError("not a SonicSphere filter pack")
    version, hlen, _ = struct.unpack("<III", data[4:16])
    if version != VERSION:
        raise ValueError(f"unsupported pack version {version}")
    header = json.loads(data[16:16 + hlen].decode("utf-8"))
    base = 16 + hlen
    arrays = {}
    for name, m in header["arrays"].items():
        dt = {"float32": "<f4", "float64": "<f8", "uint32": "<u4", "uint8": "u1"}[m["dtype"]]
        start = base + m["offset"]
        arrays[name] = np.frombuffer(data[start:start + m["byteLength"]], dtype=dt).reshape(m["shape"])
    return header, arrays


def get_pack(dataset_id: str | None = None, subject: str | None = None, fs: float | None = None,
             settings: Settings | None = None) -> tuple[bytes, str]:
    """Return (pack bytes, etag), building and caching the pack on first use."""
    settings = settings or get_settings()
    ds = get_dataset(dataset_id, subject, settings)
    fs = float(fs or ds.fs)
    etag = f"{ds.dataset_id}-{ds.subject}-{ds.sha256[:12]}-{int(round(fs))}-{settings.rendering.fingerprint()}-v{VERSION}"
    path = Path(settings.cache_dir) / f"{etag}.ssfp"
    if path.is_file():
        return path.read_bytes(), etag
    data = build_pack(get_prepared(ds.dataset_id, ds.subject, fs, settings), settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return data, etag
