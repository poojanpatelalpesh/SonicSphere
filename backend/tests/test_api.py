import io
import json

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from sonicsphere.filter_pack import parse_pack


@pytest.fixture(scope="module")
def client(cipic):
    from sonicsphere.main import app
    return TestClient(app)


def test_health_and_datasets(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and "cipic" in r.json()["datasets"]
    ds = client.get("/api/datasets").json()
    cipic = next(d for d in ds if d["id"] == "cipic")
    assert cipic["listed_subjects"] == 45 and "165" in cipic["downloaded_subjects"]


def test_dataset_info(client):
    info = client.get("/api/datasets/cipic", params={"subject": "165"}).json()
    assert info["sample_rate"] == 44100 and info["hrir_length"] == 200 and info["num_directions"] == 1250
    assert "Regents" in info["license_text"] and info["sofa_conventions"] == "SimpleFreeFieldHRIR"
    assert any(s["id"] == "165" and s["downloaded"] for s in info["subjects"])
    assert "left" in info["coordinates"]["azimuth"]


def test_unknown_dataset_and_subject(client):
    assert client.get("/api/datasets/nope").status_code == 400
    r = client.get("/api/datasets/cipic", params={"subject": "999"})
    assert r.status_code == 404 and "download_dataset" in r.json()["detail"]


def test_directions(client):
    d = client.get("/api/datasets/cipic/subjects/165/directions").json()
    assert len(d["azimuth"]) == 1250 and len(d["polar"]) == 1250


def test_filter_pack_roundtrip_and_etag(client, prepared):
    r = client.get("/api/datasets/cipic/subjects/165/filters")
    assert r.status_code == 200 and r.headers["content-type"] == "application/octet-stream"
    header, arrays = parse_pack(r.content)
    pf = prepared.filters
    assert header["filterLength"] == pf.filter_length and header["numDirections"] == 1250
    np.testing.assert_allclose(arrays["measured"], pf.measured, atol=1e-6)
    np.testing.assert_allclose(arrays["toa"], pf.toa, atol=1e-4)
    assert arrays["triangles"].shape == (header["numTriangles"], 3)
    r2 = client.get("/api/datasets/cipic/subjects/165/filters", headers={"If-None-Match": r.headers["etag"]})
    assert r2.status_code == 304


def test_filter_pack_resampled(client):
    r = client.get("/api/datasets/cipic/subjects/165/filters", params={"fs": 48000})
    header, arrays = parse_pack(r.content)
    assert header["fs"] == 48000 and header["filterLength"] % header["blockSize"] == 0
    assert client.get("/api/datasets/cipic/subjects/165/filters", params={"fs": 10}).status_code == 422


def test_prepare(client):
    p = client.post("/api/datasets/cipic/subjects/165/prepare").json()
    assert p["filter_length"] == 256 and p["num_gap_triangles"] > 0
    assert p["align_roundtrip_nmse_db_median"] < -60
    assert p["itd_range_us"][0] < -500 < 500 < p["itd_range_us"][1]


def test_hrir_endpoint(client):
    h = client.get("/api/hrir", params={"azimuth": 0, "elevation": 0}).json()
    assert h["status"] == "measured" and len(h["left"]) == 256
    h = client.get("/api/hrir", params={"azimuth": 37, "elevation": 12}).json()
    assert h["status"] == "interpolated" and abs(sum(h["weights"]) - 1) < 1e-9
    assert h["itd_us"] > 100              # source on the left -> right ear lags
    h = client.get("/api/hrir", params={"azimuth": 0, "elevation": -85}).json()
    assert h["status"] == "gap_fill"
    assert client.get("/api/hrir", params={"azimuth": 0, "elevation": 0, "method": "x"}).status_code == 422


def test_holdout_endpoint(client):
    h = client.get("/api/analysis/holdout", params={"azimuth": 50, "elevation": 20}).json()
    m = h["metrics"]
    assert m["barycentric_aligned"]["lsd_db"] < m["barycentric_raw"]["lsd_db"]
    assert len(h["measured"]["left"]) == 256


def test_sounds(client):
    s = client.get("/api/sounds").json()
    ids = {x["id"] for x in s}
    assert {"helicopter", "ak47", "guitar"} <= ids
    assert all("license" in x and x["duration_s"] > 0 for x in s)
    r = client.get("/api/sounds/helicopter")
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    assert client.get("/api/sounds/nope").status_code == 404


def _wav(x, fs=44100):
    b = io.BytesIO()
    sf.write(b, x, fs, format="WAV", subtype="PCM_16")
    return b.getvalue()


def test_render_static_upload_left_right(client, rng):
    x = 0.3 * rng.standard_normal(8000)
    for az, louder in ((90, 0), (270, 1)):
        r = client.post("/api/render", data={"params": json.dumps({"azimuth": az, "elevation": 0})},
                        files={"file": ("n.wav", _wav(x, 48000), "audio/wav")})
        assert r.status_code == 200, r.text
        y, fs = sf.read(io.BytesIO(r.content))
        info = json.loads(r.headers["x-render-info"])
        assert fs == 44100 and info["input_fs"] == 48000 and y.shape[1] == 2
        e = np.sum(y ** 2, axis=0)
        assert e[louder] > 3 * e[1 - louder]


def test_render_trajectory_builtin(client):
    pts = [[1.5 * np.sin(a), 0.0, -1.5 * np.cos(a)] for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    params = {"mode": "trajectory", "sound_id": "guitar", "points": pts, "duration_s": 2.0, "render_seconds": 2.0,
              "method": "partitioned"}
    r = client.post("/api/render", data={"params": json.dumps(params)})
    assert r.status_code == 200, r.text
    info = json.loads(r.headers["x-render-info"])
    assert info["mode"] == "trajectory" and info["filter_updates"] > 100
    y, _ = sf.read(io.BytesIO(r.content))
    assert np.max(np.abs(y)) <= 1.0 and y.shape[0] >= 2 * 44100


def test_render_validation(client):
    r = client.post("/api/render", data={"params": json.dumps({"mode": "trajectory", "sound_id": "guitar"})})
    assert r.status_code == 422
    r = client.post("/api/render", data={"params": "{\"duration_s\": -1}"})
    assert r.status_code == 422
    r = client.post("/api/render", data={"params": "{}"}, files={"file": ("x.wav", b"junk" * 100, "audio/wav")})
    assert r.status_code == 422 and "decode" in r.json()["detail"]


def test_results_endpoints(client):
    for path in ("/api/evaluation/interpolation", "/api/benchmarks"):
        r = client.get(path)
        assert r.status_code in (200, 404)
