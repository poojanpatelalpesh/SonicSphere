import h5py
import numpy as np
import pytest

from sonicsphere.dataset_loader import DatasetError, describe_grid, hrtf_to_hrir, load_sofa


def test_cipic_basic_properties(cipic):
    assert cipic.fs == 44100.0
    assert cipic.ir.shape == (1250, 2, 200)
    assert cipic.num_directions == 1250 and cipic.hrir_length == 200
    assert np.all(np.isfinite(cipic.ir))
    np.testing.assert_allclose(np.linalg.norm(cipic.vectors, axis=1), 1.0, atol=1e-12)
    assert "Regents of the University of California" in cipic.attributes.get("License", "")
    assert len(cipic.sha256) == 64


def test_cipic_grid_is_interaural_polar(cipic):
    lat, pol = cipic.interaural_polar()
    expected_lat = [-80, -65, -55] + list(range(-45, 50, 5)) + [55, 65, 80]
    np.testing.assert_allclose(sorted(set(np.round(lat, 6))), expected_lat, atol=1e-6)
    pols = np.unique(np.round(pol, 4))
    assert pols.size == 50
    np.testing.assert_allclose(pols, -45 + 5.625 * np.arange(50), atol=1e-4)
    g = describe_grid(cipic)
    assert g["num_lateral_angles"] == 25 and g["num_polar_angles"] == 50


def test_left_ear_is_channel_zero(cipic):
    # A source on the left (az 90 region) must be louder and earlier in channel 0.
    i = int(np.argmax(cipic.vectors @ np.array([0, 1.0, 0])))
    e = np.sum(cipic.ir[i] ** 2, axis=1)
    assert e[0] > 10 * e[1]
    assert np.argmax(np.abs(cipic.ir[i, 0])) < np.argmax(np.abs(cipic.ir[i, 1]))
    j = int(np.argmax(cipic.vectors @ np.array([0, -1.0, 0])))
    e = np.sum(cipic.ir[j] ** 2, axis=1)
    assert e[1] > 10 * e[0]


def _write_sofa(path, ir, pos, fs=44100.0, conventions="SimpleFreeFieldHRIR", receivers=((0, .09, 0), (0, -.09, 0)),
                pos_type="spherical", extra=None):
    with h5py.File(path, "w") as f:
        f.attrs["Conventions"] = np.bytes_("SOFA")
        f.attrs["SOFAConventions"] = np.bytes_(conventions)
        if ir is not None:
            f["Data.IR"] = ir
        f["Data.SamplingRate"] = np.array([fs])
        f["Data.SamplingRate"].attrs["Units"] = np.bytes_("hertz")
        d = f.create_dataset("SourcePosition", data=pos)
        d.attrs["Type"] = np.bytes_(pos_type)
        d.attrs["Units"] = np.bytes_("degree, degree, metre")
        f["ListenerView"] = np.array([[1.0, 0, 0]])
        f["ListenerUp"] = np.array([[0, 0, 1.0]])
        f["ReceiverPosition"] = np.array(receivers, float)[:, :, None]
        for k, v in (extra or {}).items():
            f[k] = v


def _grid(n=6):
    az = np.repeat(np.arange(0, 360, 60.0), 3)
    el = np.tile([-30.0, 0.0, 40.0], 6)
    pos = np.stack([az, el, np.ones_like(az)], axis=1)
    return np.vstack([pos, [[0, 90, 1], [0, -90, 1]]])


def test_load_synthetic_sofa_and_receiver_order(tmp_path, rng):
    pos = _grid()
    ir = rng.standard_normal((len(pos), 2, 64))
    p = tmp_path / "a.sofa"
    _write_sofa(p, ir, pos, receivers=((0, -.09, 0), (0, .09, 0)))  # right ear stored first
    ds = load_sofa(p)
    np.testing.assert_allclose(ds.ir[:, 0], ir[:, 1])   # reordered: channel 0 = left
    assert any("reordered" in n for n in ds.notes)


def test_listener_view_rotation(tmp_path, rng):
    pos = _grid()
    p = tmp_path / "b.sofa"
    _write_sofa(p, rng.standard_normal((len(pos), 2, 32)), pos)
    with h5py.File(p, "a") as f:
        f["ListenerView"][...] = np.array([[0, 1.0, 0]])  # listener looks along global +y
    ds = load_sofa(p)
    # A source at global azimuth 90 (on +y) is straight ahead for this listener.
    i = int(np.argmin(np.abs(pos[:, 0] - 60) + np.abs(pos[:, 1])))  # az 60 el 0
    assert np.isclose(ds.azimuth[i], 330.0)


@pytest.mark.parametrize("mutate,msg", [
    (lambda f: f.attrs.__setitem__("Conventions", np.bytes_("NOPE")), "Conventions"),
    (lambda f: f.__delitem__("SourcePosition"), "SourcePosition"),
    (lambda f: f.__delitem__("Data.IR"), "Data.IR"),
])
def test_malformed_files_raise(tmp_path, rng, mutate, msg):
    pos = _grid()
    p = tmp_path / "c.sofa"
    _write_sofa(p, rng.standard_normal((len(pos), 2, 32)), pos)
    with h5py.File(p, "a") as f:
        mutate(f)
    with pytest.raises(DatasetError, match=msg):
        load_sofa(p)


def test_nan_and_shape_errors(tmp_path, rng):
    pos = _grid()
    ir = rng.standard_normal((len(pos), 2, 32))
    ir[3, 1, 5] = np.nan
    p = tmp_path / "d.sofa"
    _write_sofa(p, ir, pos)
    with pytest.raises(DatasetError, match="NaN"):
        load_sofa(p)
    p2 = tmp_path / "e.sofa"
    _write_sofa(p2, rng.standard_normal((len(pos) - 1, 2, 32)), pos)
    with pytest.raises(DatasetError, match="rows"):
        load_sofa(p2)
    p3 = tmp_path / "f.sofa"
    _write_sofa(p3, rng.standard_normal((len(pos), 3, 32)), pos)
    with pytest.raises(DatasetError, match="2 receivers"):
        load_sofa(p3)


def test_missing_and_non_hdf5_files(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        load_sofa(tmp_path / "missing.sofa")
    bad = tmp_path / "bad.sofa"
    bad.write_bytes(b"RIFF....WAVE")
    with pytest.raises(DatasetError, match="HDF5"):
        load_sofa(bad)


def test_missing_subject_has_helpful_message(settings):
    from sonicsphere.dataset_loader import load_dataset
    with pytest.raises(DatasetError, match="download_dataset"):
        load_dataset(settings.dataset("cipic"), "999")


def test_hrtf_to_hrir_roundtrip(rng):
    fs = 48000.0
    for n in (256, 255):
        h = rng.standard_normal((4, 2, n))
        spec = np.fft.rfft(h, axis=-1)
        freqs = np.fft.rfftfreq(n, 1 / fs)
        ir, fs2, notes = hrtf_to_hrir(spec.real, spec.imag, freqs, fs)
        assert fs2 == fs and ir.shape[-1] == n
        np.testing.assert_allclose(ir, h, atol=1e-12)
    # Imaginary DC violates conjugate symmetry: reported and discarded.
    spec = np.fft.rfft(rng.standard_normal(64))
    spec[0] += 1j
    _, _, notes = hrtf_to_hrir(spec.real, spec.imag, np.fft.rfftfreq(64, 1 / fs), fs)
    assert any("DC" in n for n in notes)
    with pytest.raises(DatasetError):
        hrtf_to_hrir(spec.real, spec.imag, np.r_[0, np.cumsum(np.random.rand(32) + 1)], fs)


def test_mit_kemar_through_same_adapter(settings):
    from sonicsphere.dataset_loader import get_dataset
    try:
        ds = get_dataset("mit_kemar", "normal_pinna", settings)
    except DatasetError:
        pytest.skip("MIT KEMAR not downloaded")
    assert ds.ir.shape[1] == 2 and ds.fs == 44100.0
    i = int(np.argmax(ds.vectors @ np.array([0, 1.0, 0])))
    e = np.sum(ds.ir[i] ** 2, axis=1)
    assert e[0] > e[1]
