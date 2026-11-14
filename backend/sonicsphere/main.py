"""SonicSphere HTTP API (FastAPI).

Start:  uvicorn sonicsphere.main:app --reload      (from backend/)

The browser calls this API to obtain dataset metadata and a binary filter pack once per
subject; real-time rendering then happens entirely in the browser. Heavier work
(preprocessing, reference renders, evaluation, benchmarks) runs here.
"""

from __future__ import annotations

import json
import time

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from . import __version__
from .audio_processing import (AudioInputError, apply_stereo_policy, decode_audio, ensure_min_length,
                               normalize_sources, resample, to_wav_bytes)
from .audio_renderer import ListenerPose, distance_gain, render_static, render_trajectory
from .config import ConfigError, get_settings
from .coordinate_system import cart_to_interaural_polar, cart_to_sph, head_to_world, sph_to_cart
from .dataset_loader import DatasetError, describe_grid, get_dataset
from .filter_pack import get_pack, get_prepared
from .hrir_interpolation import METHODS, HRIRInterpolator, build_triangulation
from .hrir_preprocessing import estimate_toa, reconstruct
from .interpolation_eval import _subset, lsd_db, nmse_db
from .schemas import (CoordinateConventions, DatasetInfo, DatasetSummary, Directions, HoldoutResponse, HRIRResponse,
                      PrepareResponse, RenderParams, Spectrum, SubjectInfo)
from .trajectory import Trajectory

settings = get_settings()
app = FastAPI(title="SonicSphere API", version=__version__,
              description="HRIR dataset access, filter preparation and reference binaural rendering.")
app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["*"],
                   allow_headers=["*"], expose_headers=["X-Render-Info", "ETag"])
app.add_middleware(GZipMiddleware, minimum_size=2048)

COORDS = CoordinateConventions(
    frame="Head frame (SOFA): x = front, y = left ear, z = up; origin at the centre of the head.",
    azimuth="Azimuth in [0, 360): 0 = front, 90 = left, 180 = back, 270 = right (counter-clockwise from above).",
    elevation="Elevation in [-90, 90]: 0 = horizontal plane, +90 = above the head.",
    interaural_polar="Lateral angle = asin(y), positive to the left; polar angle = atan2(z, x): 0 front, 90 above, 180 behind.",
    dataset_native=("CIPIC was measured on an interaural-polar grid: 25 lateral angles (-80..80 deg) x 50 polar "
                    "angles (-45..230.625 deg in 5.625 deg steps) at 1 m. The SOFA release stores the same "
                    "positions as SOFA spherical azimuth/elevation; SonicSphere reads those and converts them."),
    world_frame=("3D scene: right-handed, +Y up, metres. At yaw 0 the listener faces -Z, so world -X is the "
                 "listener's left. Yaw is positive to the left (counter-clockwise from above), pitch positive nose-up."),
)


def _err(status: int, msg: str):
    raise HTTPException(status_code=status, detail=msg)


@app.exception_handler(DatasetError)
async def _dataset_error(_: Request, exc: DatasetError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(ConfigError)
async def _config_error(_: Request, exc: ConfigError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(AudioInputError)
async def _audio_error(_: Request, exc: AudioInputError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


# ------------------------------------------------------------------ general

@app.get("/api/health")
def health():
    return {"status": "ok", "version": __version__,
            "datasets": {d: c.available_subjects() for d, c in settings.datasets.items()}}


@app.get("/api/config")
def config():
    r = settings.rendering
    return {"rendering": r.__dict__, "default_dataset": settings.default_dataset,
            "upload": {"max_bytes": settings.upload_max_bytes, "max_duration_s": settings.upload_max_duration_s,
                       "min_duration_s": settings.upload_min_duration_s},
            "interpolation_methods": sorted(METHODS)}


# ------------------------------------------------------------------ datasets

@app.get("/api/datasets", response_model=list[DatasetSummary])
def datasets():
    return [DatasetSummary(id=c.id, name=c.name, default_subject=c.default_subject,
                           downloaded_subjects=c.available_subjects(), listed_subjects=len(c.subjects))
            for c in settings.datasets.values()]


@app.get("/api/datasets/{dataset_id}", response_model=DatasetInfo)
def dataset_info(dataset_id: str, subject: str | None = None):
    cfg = settings.dataset(dataset_id)
    ds = get_dataset(dataset_id, subject, settings)
    downloaded = set(cfg.available_subjects())
    subj = [SubjectInfo(id=s, downloaded=s in downloaded, note=cfg.subject_notes.get(s, ""))
            for s in (list(cfg.subjects) + [s for s in downloaded if s not in cfg.subjects])]
    attrs = {k: v[:400] for k, v in ds.attributes.items()}
    return DatasetInfo(
        id=cfg.id, name=cfg.name, subject=ds.subject, homepage=cfg.homepage, download_url=cfg.download_base_url,
        reference=cfg.reference or ds.attributes.get("References", ""), license_summary=cfg.license_summary,
        license_text=ds.attributes.get("License", ""), sofa_conventions=ds.conventions, sample_rate=ds.fs,
        hrir_length=ds.hrir_length, num_directions=ds.num_directions, data_type=ds.attributes.get("DataType", "FIR"),
        subjects=subj, grid=describe_grid(ds), coordinates=COORDS, notes=ds.notes, sha256=ds.sha256, attributes=attrs)


@app.get("/api/datasets/{dataset_id}/subjects/{subject}/directions", response_model=Directions)
def directions(dataset_id: str, subject: str):
    ds = get_dataset(dataset_id, subject, settings)
    lat, pol = cart_to_interaural_polar(ds.vectors)
    return Directions(azimuth=ds.azimuth.round(4).tolist(), elevation=ds.elevation.round(4).tolist(),
                      lateral=lat.round(4).tolist(), polar=pol.round(4).tolist())


def _check_fs(fs: float | None) -> float | None:
    if fs is not None and not (8000 <= fs <= 192000):
        _err(422, "fs must be between 8000 and 192000 Hz")
    return fs


@app.get("/api/datasets/{dataset_id}/subjects/{subject}/filters")
def filters(dataset_id: str, subject: str, request: Request, fs: float | None = Query(None)):
    """Binary filter pack (see filter_pack.py for the layout)."""
    data, etag = get_pack(dataset_id, subject, _check_fs(fs), settings)
    if request.headers.get("if-none-match") == f'"{etag}"':
        return Response(status_code=304)
    return Response(content=data, media_type="application/octet-stream",
                    headers={"ETag": f'"{etag}"', "Cache-Control": "public, max-age=86400"})


@app.post("/api/datasets/{dataset_id}/subjects/{subject}/prepare", response_model=PrepareResponse)
def prepare(dataset_id: str, subject: str, fs: float | None = Query(None)):
    t0 = time.perf_counter()
    prep = get_prepared(dataset_id, subject, _check_fs(fs), settings)
    data, etag = get_pack(dataset_id, subject, fs, settings)
    pf, tri = prep.filters, prep.triangulation
    rec = reconstruct(pf.aligned, pf.toa, pf.margin, pf.filter_length)
    e = nmse_db(rec, pf.measured)
    itd = (pf.toa[:, 1] - pf.toa[:, 0]) / pf.fs * 1e6
    return PrepareResponse(
        dataset=dataset_id, subject=prep.dataset.subject, fs=pf.fs, filter_length=pf.filter_length,
        align_margin=pf.margin, toa_range_samples=[float(pf.toa.min()), float(pf.toa.max())],
        itd_range_us=[float(itd.min()), float(itd.max())], num_triangles=tri.num_triangles,
        num_gap_triangles=int(tri.is_gap.sum()), num_boundary_edges=int(tri.boundary_edges.shape[0]),
        max_valid_edge_deg=float(tri.max_edge_deg[~tri.is_gap].max()), truncated_energy_db=pf.truncated_energy_db,
        align_roundtrip_nmse_db_median=float(np.median(e)), align_roundtrip_nmse_db_max=float(e.max()),
        gain=pf.gain, pack_bytes=len(data), etag=etag, seconds=time.perf_counter() - t0)


# ------------------------------------------------------------------ analysis

_INTERP_CACHE: dict = {}


def _interp(dataset_id, subject, method):
    prep = get_prepared(dataset_id, subject, None, settings)
    key = (id(prep), method)
    if key not in _INTERP_CACHE:
        r = settings.rendering
        _INTERP_CACHE[key] = HRIRInterpolator(
            prep.filters, method, r.gap_edge_threshold_deg, r.measured_snap_deg,
            triangulation=prep.triangulation if METHODS[method][0] == "barycentric" else None,
            direction_quantum_deg=r.direction_quantum_deg)
    return prep, _INTERP_CACHE[key]


def _spectrum(h: np.ndarray, fs: float, nfft: int = 1024) -> Spectrum:
    H = np.fft.rfft(h, nfft, axis=-1)
    f = np.fft.rfftfreq(nfft, 1 / fs)
    db = 20 * np.log10(np.maximum(np.abs(H), 1e-9))
    return Spectrum(freq_hz=f.round(2).tolist(), left_db=db[0].round(3).tolist(), right_db=db[1].round(3).tolist())


@app.get("/api/hrir", response_model=HRIRResponse)
def hrir(azimuth: float = Query(..., ge=-360, le=720), elevation: float = Query(..., ge=-90, le=90),
         dataset: str | None = None, subject: str | None = None, method: str | None = None):
    method = method or settings.rendering.interpolation
    if method not in METHODS:
        _err(422, f"Unknown method '{method}'")
    cfg = settings.dataset(dataset)
    prep, it = _interp(cfg.id, subject or cfg.default_subject, method)
    pf = prep.filters
    res = it.query_sph(azimuth % 360, elevation)
    raz, rel, _ = cart_to_sph(res.w.rendered_direction)
    ds = prep.dataset
    neigh = [{"index": int(i), "weight": float(w), "azimuth": float(ds.azimuth[i]), "elevation": float(ds.elevation[i])}
             for i, w in zip(res.w.indices, res.w.weights)]
    return HRIRResponse(
        dataset=cfg.id, subject=ds.subject, method=method, requested={"azimuth": azimuth % 360, "elevation": elevation},
        rendered={"azimuth": float(raz), "elevation": float(rel)}, status=res.w.status,
        indices=[int(i) for i in res.w.indices], weights=[float(w) for w in res.w.weights], neighbours=neigh,
        toa_samples=[float(t) for t in res.toa], itd_us=float((res.toa[1] - res.toa[0]) / pf.fs * 1e6), fs=pf.fs,
        left=res.hrir[0].round(7).tolist(), right=res.hrir[1].round(7).tolist(), spectrum=_spectrum(res.hrir, pf.fs),
        extra={k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in res.w.extra.items()}
        | {"nearest_angle_deg": res.w.nearest_angle_deg, "max_edge_deg": res.w.max_edge_deg})


@app.get("/api/analysis/holdout", response_model=HoldoutResponse)
def holdout(index: int | None = None, azimuth: float | None = None, elevation: float | None = None,
            dataset: str | None = None, subject: str | None = None):
    """Leave-one-out: withhold one measured direction, estimate it from the others with
    every method, and compare with the true measurement."""
    cfg = settings.dataset(dataset)
    prep = get_prepared(cfg.id, subject or cfg.default_subject, None, settings)
    pf, ds = prep.filters, prep.dataset
    if index is None:
        target = sph_to_cart(azimuth if azimuth is not None else 50.0, elevation if elevation is not None else 20.0)
        index = int(np.argmax(pf.vectors @ target))
    if not (0 <= index < len(pf.vectors)):
        _err(422, "index out of range")
    keep = np.setdiff1d(np.arange(len(pf.vectors)), [index])
    train = _subset(pf, keep)
    r = settings.rendering
    tri = build_triangulation(train.vectors, r.gap_edge_threshold_deg)
    truth = pf.measured[index]
    est, metrics = {}, {}
    for m in ("nearest", "barycentric_raw", "barycentric_aligned", "idw_aligned", "barycentric_minphase"):
        it = HRIRInterpolator(train, m, r.gap_edge_threshold_deg, 0.0,
                              triangulation=tri if METHODS[m][0] == "barycentric" else None)
        q = it.query(pf.vectors[index])
        toa = estimate_toa(q.hrir, r.onset_threshold_db, r.onset_upsample)
        est[m] = {"left": q.hrir[0].round(7).tolist(), "right": q.hrir[1].round(7).tolist(),
                  "spectrum": _spectrum(q.hrir, pf.fs).model_dump(), "status": q.w.status}
        metrics[m] = {"lsd_db": float(np.mean(lsd_db(q.hrir, truth, pf.fs))),
                      "nmse_db": float(np.mean(nmse_db(q.hrir, truth))),
                      "itd_err_us": float(abs((toa[1] - toa[0]) - (pf.toa[index, 1] - pf.toa[index, 0])) / pf.fs * 1e6)}
    lat, pol = cart_to_interaural_polar(pf.vectors[index])
    return HoldoutResponse(
        dataset=cfg.id, subject=ds.subject, index=index,
        direction={"azimuth": float(ds.azimuth[index]), "elevation": float(ds.elevation[index]),
                   "lateral": float(lat), "polar": float(pol)},
        fs=pf.fs, measured={"left": truth[0].round(7).tolist(), "right": truth[1].round(7).tolist(),
                            "spectrum": _spectrum(truth, pf.fs).model_dump()},
        estimates=est, metrics=metrics)


def _read_results(name: str):
    path = settings.results_dir / name
    if not path.is_file():
        return None
    return json.loads(path.read_text())


@app.get("/api/evaluation/interpolation")
def interpolation_results():
    res = _read_results("interpolation_eval.json")
    if res is None:
        _err(404, "No evaluation results yet. Run: python -m sonicsphere.interpolation_eval")
    return res


@app.post("/api/evaluation/interpolation/run")
def run_interpolation_eval(subjects: list[str] | None = Query(None)):
    from .interpolation_eval import results_path, run
    res = run(subjects)
    results_path().parent.mkdir(parents=True, exist_ok=True)
    results_path().write_text(json.dumps(res, indent=2))
    return res


@app.get("/api/benchmarks")
def benchmarks():
    res = _read_results("benchmark.json")
    if res is None:
        _err(404, "No benchmark results yet. Run: python -m sonicsphere.benchmark")
    return res


@app.post("/api/benchmarks/run")
def run_benchmarks(quick: bool = True):
    from .benchmark import results_path, run
    res = run(quick)
    results_path().parent.mkdir(parents=True, exist_ok=True)
    results_path().write_text(json.dumps(res, indent=2))
    return res


# ------------------------------------------------------------------ sounds

def _manifest() -> list[dict]:
    path = settings.sounds_dir / "manifest.json"
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/sounds")
def sounds():
    out = []
    for s in _manifest():
        if (settings.sounds_dir / s["file"]).is_file():
            out.append(s | {"url": f"/api/sounds/{s['id']}"})
    return out


@app.get("/api/sounds/{sound_id}")
def sound_file(sound_id: str):
    for s in _manifest():
        if s["id"] == sound_id:
            path = settings.sounds_dir / s["file"]
            if path.is_file():
                return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "public, max-age=3600"})
    _err(404, f"Unknown sound '{sound_id}'. Run: python -m sonicsphere.generate_sounds")


# ------------------------------------------------------------------ reference rendering

@app.post("/api/render")
async def render(params: str = Form("{}"), file: UploadFile | None = File(None)):
    """Reference binaural render (Python). Returns a 16-bit stereo WAV; details of the
    render are returned as JSON in the X-Render-Info header."""
    try:
        p = RenderParams.model_validate_json(params or "{}")
    except Exception as exc:
        _err(422, f"Invalid params: {exc}")
    r = settings.rendering
    cfg = settings.dataset(p.dataset)
    method = p.interpolation or r.interpolation
    prep, it = _interp(cfg.id, p.subject or cfg.default_subject, method)
    pf = prep.filters
    if file is not None:
        raw = await file.read()
        origin = file.filename or "upload"
    else:
        sid = p.sound_id or "helicopter"
        entry = next((s for s in _manifest() if s["id"] == sid), None)
        if entry is None:
            _err(404, f"Unknown sound '{sid}'")
        raw = (settings.sounds_dir / entry["file"]).read_bytes()
        origin = sid
    x, fs_in = decode_audio(raw, settings.upload_max_bytes, settings.upload_max_duration_s,
                            settings.upload_min_duration_s)
    x = resample(x, fs_in, pf.fs)
    src = apply_stereo_policy(x, pf.fs, p.stereo_policy, p.stereo_width_deg or r.stereo_pair_width_deg)
    signals = [ensure_min_length(s, r.block_size) for s in src.signals]
    norm_gain = 1.0
    if p.normalize:
        signals, norm_gain = normalize_sources(signals, r.target_rms_db, r.peak_ceiling_db)
    pose = ListenerPose(np.array(p.listener_position, float), p.listener_yaw_deg, p.listener_pitch_deg)
    t0 = time.perf_counter()
    info: dict = {"source": origin, "fs": pf.fs, "input_fs": fs_in, "stereo_policy": src.policy,
                  "stereo_analysis": src.analysis, "normalisation_gain": norm_gain, "interpolation": method,
                  "dataset": cfg.id, "subject": prep.dataset.subject}
    if p.mode == "static":
        d_head = sph_to_cart(p.azimuth % 360, p.elevation) * max(p.distance, 1e-3)
        g = pf.gain * distance_gain(p.distance, p.distance_attenuation)
        out = None
        statuses = []
        for sig, off in zip(signals, src.azimuth_offsets_deg):
            q = it.query_sph((p.azimuth + off) % 360, p.elevation)
            statuses.append(q.w.status)
            y = render_static(sig, q.hrir, "partitioned" if p.method == "partitioned" else p.method, g)
            out = y if out is None else out + y
        world = head_to_world(d_head, pose.position, pose.yaw_deg, pose.pitch_deg)
        info.update({"mode": "static", "convolution": p.method, "status": statuses,
                     "world_position": world.round(4).tolist()})
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > r.limiter_threshold:          # offline: scale instead of limiting
            out *= r.limiter_threshold / peak
            info["output_scaled_by"] = r.limiter_threshold / peak
    else:
        if not p.points:
            _err(422, "trajectory mode needs at least one point")
        traj = Trajectory.from_points(np.array(p.points, float), p.duration_s, p.loop)
        secs = p.render_seconds or max(len(signals[0]) / pf.fs, p.duration_s if not p.loop else 0)
        n = int(min(secs, settings.upload_max_duration_s) * pf.fs)
        sigs = [np.resize(s, n) if p.loop else ensure_min_length(s, n)[:n] for s in signals]
        engine = "partitioned" if p.method in ("fft", "partitioned") else "direct"
        out, eng = render_trajectory(sigs, src.azimuth_offsets_deg, it, traj, pose, gain=pf.gain, method=engine,
                                     distance_attenuation=p.distance_attenuation,
                                     limiter_threshold=r.limiter_threshold, limiter_release_ms=r.limiter_release_ms)
        info.update({"mode": "trajectory", "convolution": engine, "blocks": eng.stats.blocks,
                     "filter_updates": eng.stats.filter_updates, "crossfades": eng.stats.crossfades,
                     "limiter_min_gain": eng.limiter.min_gain, "trajectory_length_m": traj.length()})
    info["render_seconds"] = time.perf_counter() - t0
    info["output_samples"] = int(out.shape[0])
    wav = to_wav_bytes(out, pf.fs)
    return Response(content=wav, media_type="audio/wav",
                    headers={"X-Render-Info": json.dumps(info, default=float, ensure_ascii=True),
                             "Content-Disposition": f'attachment; filename="sonicsphere_{origin}_{p.mode}.wav"'})


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run("sonicsphere.main:app", host=settings.host, port=settings.port, reload=False)
