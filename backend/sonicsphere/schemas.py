"""Typed request / response models for the HTTP API."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

InterpMethod = Literal["nearest", "barycentric_aligned", "barycentric_raw", "idw_aligned", "idw_raw",
                       "barycentric_minphase"]
StereoPolicy = Literal["mid", "left", "right", "stereo_pair"]


class SubjectInfo(BaseModel):
    id: str
    downloaded: bool
    note: str = ""


class DatasetSummary(BaseModel):
    id: str
    name: str
    default_subject: str
    downloaded_subjects: list[str]
    listed_subjects: int


class CoordinateConventions(BaseModel):
    frame: str
    azimuth: str
    elevation: str
    interaural_polar: str
    dataset_native: str
    world_frame: str


class DatasetInfo(BaseModel):
    id: str
    name: str
    subject: str
    homepage: str
    download_url: str
    reference: str
    license_summary: str
    license_text: str
    sofa_conventions: str
    sample_rate: float
    hrir_length: int
    num_directions: int
    data_type: str
    subjects: list[SubjectInfo]
    grid: dict
    coordinates: CoordinateConventions
    notes: list[str]
    sha256: str
    attributes: dict[str, str]


class Directions(BaseModel):
    azimuth: list[float]
    elevation: list[float]
    lateral: list[float]
    polar: list[float]


class PrepareResponse(BaseModel):
    dataset: str
    subject: str
    fs: float
    filter_length: int
    align_margin: int
    toa_range_samples: list[float]
    itd_range_us: list[float]
    num_triangles: int
    num_gap_triangles: int
    num_boundary_edges: int
    max_valid_edge_deg: float
    truncated_energy_db: float
    align_roundtrip_nmse_db_median: float
    align_roundtrip_nmse_db_max: float
    gain: float
    pack_bytes: int
    etag: str
    seconds: float


class Spectrum(BaseModel):
    freq_hz: list[float]
    left_db: list[float]
    right_db: list[float]


class HRIRResponse(BaseModel):
    dataset: str
    subject: str
    method: str
    requested: dict
    rendered: dict
    status: str
    indices: list[int]
    weights: list[float]
    neighbours: list[dict]
    toa_samples: list[float]
    itd_us: float
    fs: float
    left: list[float]
    right: list[float]
    spectrum: Spectrum
    extra: dict = {}


class HoldoutResponse(BaseModel):
    dataset: str
    subject: str
    index: int
    direction: dict
    fs: float
    measured: dict
    estimates: dict
    metrics: dict


class RenderParams(BaseModel):
    dataset: Optional[str] = None
    subject: Optional[str] = None
    sound_id: Optional[str] = Field(None, description="built-in sound id; ignored if a file is uploaded")
    mode: Literal["static", "trajectory"] = "static"
    azimuth: float = 90.0
    elevation: float = 0.0
    distance: float = 1.5
    points: list[list[float]] = Field(default_factory=list, description="world-space control points [[x,y,z],...]")
    duration_s: float = 4.0
    loop: bool = True
    render_seconds: Optional[float] = None
    listener_position: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    listener_yaw_deg: float = 0.0
    listener_pitch_deg: float = 0.0
    method: Literal["direct", "fft", "partitioned"] = "direct"
    interpolation: Optional[InterpMethod] = None
    stereo_policy: StereoPolicy = "mid"
    stereo_width_deg: Optional[float] = None
    distance_attenuation: bool = True
    normalize: bool = True

    @field_validator("points")
    @classmethod
    def _points(cls, v):
        for p in v:
            if len(p) != 3:
                raise ValueError("each point must be [x, y, z]")
        if len(v) > 4096:
            raise ValueError("at most 4096 trajectory points")
        return v

    @field_validator("duration_s")
    @classmethod
    def _dur(cls, v):
        if not (0.1 <= v <= 600):
            raise ValueError("duration_s must be between 0.1 and 600")
        return v

    @field_validator("listener_position")
    @classmethod
    def _lp(cls, v):
        if len(v) != 3:
            raise ValueError("listener_position must be [x, y, z]")
        return v
