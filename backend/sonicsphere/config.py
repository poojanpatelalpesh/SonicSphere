"""Configuration loading.

The configuration lives in a TOML file. Resolution order:

1. ``$SONICSPHERE_CONFIG`` if set,
2. ``backend/sonicsphere.toml`` if present,
3. ``backend/config.example.toml`` (the committed template).

Relative paths inside the file are resolved against the file's directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parent.parent


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class DatasetConfig:
    id: str
    name: str
    format: str
    directory: Path
    file_pattern: str
    default_subject: str
    subjects: tuple[str, ...]
    default_download: tuple[str, ...]
    download_base_url: str = ""
    homepage: str = ""
    reference: str = ""
    license_summary: str = ""
    subject_notes: dict[str, str] = field(default_factory=dict)

    def subject_path(self, subject: str) -> Path:
        return self.directory / self.file_pattern.format(subject=subject)

    def available_subjects(self) -> list[str]:
        """Subjects whose files exist on disk (listed subjects first, then any extra files)."""
        found = [s for s in self.subjects if self.subject_path(s).is_file()]
        prefix, _, suffix = self.file_pattern.partition("{subject}")
        if self.directory.is_dir():
            for p in sorted(self.directory.iterdir()):
                name = p.name
                if p.is_file() and name.startswith(prefix) and name.endswith(suffix):
                    sid = name[len(prefix): len(name) - len(suffix)]
                    if sid and sid not in found:
                        found.append(sid)
        return found


@dataclass(frozen=True)
class RenderingConfig:
    block_size: int = 128
    align_margin: int = 40
    onset_threshold_db: float = -20.0
    onset_upsample: int = 8
    interpolation: str = "barycentric_aligned"
    gap_edge_threshold_deg: float = 30.0
    measured_snap_deg: float = 0.1
    direction_quantum_deg: float = 0.5
    target_rms_db: float = -20.0
    peak_ceiling_db: float = -1.0
    limiter_threshold: float = 0.98
    limiter_release_ms: float = 80.0
    stereo_pair_width_deg: float = 30.0

    def fingerprint(self) -> str:
        """Hash of the parameters that influence prepared filters (used for cache keys)."""
        keys = ("block_size", "align_margin", "onset_threshold_db", "onset_upsample",
                "gap_edge_threshold_deg", "direction_quantum_deg", "measured_snap_deg", "limiter_threshold",
                "limiter_release_ms", "interpolation")
        blob = json.dumps({k: getattr(self, k) for k in keys}, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:10]


@dataclass(frozen=True)
class Settings:
    config_path: Path
    host: str
    port: int
    cors_origins: tuple[str, ...]
    cache_dir: Path
    results_dir: Path
    sounds_dir: Path
    upload_max_bytes: int
    upload_max_duration_s: float
    upload_min_duration_s: float
    rendering: RenderingConfig
    default_dataset: str
    datasets: dict[str, DatasetConfig]

    def dataset(self, dataset_id: str | None = None) -> DatasetConfig:
        did = dataset_id or self.default_dataset
        try:
            return self.datasets[did]
        except KeyError:
            raise ConfigError(f"Unknown dataset '{did}'. Configured: {sorted(self.datasets)}") from None


_VALID_INTERP = {"nearest", "barycentric_aligned", "barycentric_raw", "idw_aligned", "idw_raw",
                 "barycentric_minphase"}


def _resolve(base: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (base / path).resolve()


def load_settings(path: str | os.PathLike | None = None) -> Settings:
    if path is None:
        env = os.environ.get("SONICSPHERE_CONFIG")
        if env:
            path = env
        elif (BACKEND_DIR / "sonicsphere.toml").is_file():
            path = BACKEND_DIR / "sonicsphere.toml"
        else:
            path = BACKEND_DIR / "config.example.toml"
    cfg_path = Path(path).resolve()
    if not cfg_path.is_file():
        raise ConfigError(f"Configuration file not found: {cfg_path}")
    with cfg_path.open("rb") as fh:
        raw: dict[str, Any] = tomllib.load(fh)
    base = cfg_path.parent

    server = raw.get("server", {})
    paths = raw.get("paths", {})
    upload = raw.get("upload", {})
    rendering = RenderingConfig(**raw.get("rendering", {}))
    if rendering.interpolation not in _VALID_INTERP:
        raise ConfigError(f"rendering.interpolation must be one of {sorted(_VALID_INTERP)}")
    if rendering.block_size <= 0 or rendering.block_size & (rendering.block_size - 1):
        raise ConfigError("rendering.block_size must be a positive power of two")

    datasets: dict[str, DatasetConfig] = {}
    for did, d in raw.get("datasets", {}).items():
        try:
            datasets[did] = DatasetConfig(
                id=did,
                name=d["name"],
                format=d.get("format", "sofa"),
                directory=_resolve(base, d["directory"]),
                file_pattern=d["file_pattern"],
                default_subject=str(d["default_subject"]),
                subjects=tuple(str(s) for s in d.get("subjects", [])),
                default_download=tuple(str(s) for s in d.get("default_download", [])),
                download_base_url=d.get("download_base_url", ""),
                homepage=d.get("homepage", ""),
                reference=d.get("reference", ""),
                license_summary=d.get("license_summary", ""),
                subject_notes={str(k): str(v) for k, v in d.get("subject_notes", {}).items()},
            )
        except KeyError as exc:
            raise ConfigError(f"Dataset '{did}' is missing required key {exc}") from None
        if "{subject}" not in datasets[did].file_pattern:
            raise ConfigError(f"Dataset '{did}': file_pattern must contain '{{subject}}'")
    default_dataset = raw.get("dataset", {}).get("default", next(iter(datasets), ""))
    if default_dataset not in datasets:
        raise ConfigError(f"Default dataset '{default_dataset}' is not configured")

    return Settings(
        config_path=cfg_path,
        host=server.get("host", "127.0.0.1"),
        port=int(server.get("port", 8000)),
        cors_origins=tuple(server.get("cors_origins", [])),
        cache_dir=_resolve(base, paths.get("cache_dir", "cache")),
        results_dir=_resolve(base, paths.get("results_dir", "results")),
        sounds_dir=_resolve(base, paths.get("sounds_dir", "../assets/sounds")),
        upload_max_bytes=int(upload.get("max_bytes", 25 * 1024 * 1024)),
        upload_max_duration_s=float(upload.get("max_duration_s", 60.0)),
        upload_min_duration_s=float(upload.get("min_duration_s", 0.01)),
        rendering=rendering,
        default_dataset=default_dataset,
        datasets=datasets,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
