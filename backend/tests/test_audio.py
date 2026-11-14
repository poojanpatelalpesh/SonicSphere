import io

import numpy as np
import pytest
import soundfile as sf

from sonicsphere.audio_processing import (AudioInputError, PeakLimiter, analyse_stereo, apply_stereo_policy,
                                          decode_audio, ensure_min_length, normalize_sources, resample, to_wav_bytes)
from sonicsphere.audio_renderer import ListenerPose, distance_gain, render_moving, render_static
from sonicsphere.convolution import direct_convolve
from sonicsphere.coordinate_system import sph_to_cart
from sonicsphere.trajectory import Trajectory

LIM = dict(max_bytes=10_000_000, max_duration_s=60, min_duration_s=0.01)


def _wav(x, fs, subtype="PCM_16", fmt="WAV"):
    b = io.BytesIO()
    sf.write(b, x, fs, format=fmt, subtype=subtype)
    return b.getvalue()


# ------------------------------------------------------------------ input handling

def test_decode_mono_stereo_and_flac(rng):
    x = np.clip(0.2 * rng.standard_normal((4410, 2)), -0.99, 0.99)
    y, fs = decode_audio(_wav(x, 44100), **LIM)
    assert fs == 44100 and y.shape == (4410, 2)
    np.testing.assert_allclose(y, x, atol=1 / 32768 + 1e-6)
    y, _ = decode_audio(_wav(x[:, 0], 22050, fmt="FLAC"), **LIM)
    assert y.shape == (4410, 1)


@pytest.mark.parametrize("data,msg", [(b"", "empty"), (b"definitely not audio" * 10, "decode")])
def test_decode_rejects_bad_files(data, msg):
    with pytest.raises(AudioInputError, match=msg):
        decode_audio(data, **LIM)


def test_decode_limits(rng):
    with pytest.raises(AudioInputError, match="limit"):
        decode_audio(_wav(np.zeros(44100 * 3), 44100), max_bytes=10_000_000, max_duration_s=2, min_duration_s=0.01)
    with pytest.raises(AudioInputError, match="minimum"):
        decode_audio(_wav(np.zeros(100), 44100), **LIM)
    with pytest.raises(AudioInputError, match="MB"):
        decode_audio(_wav(np.zeros(44100), 44100), max_bytes=1000, max_duration_s=60, min_duration_s=0.01)
    with pytest.raises(AudioInputError, match="channel"):
        decode_audio(_wav(np.zeros((4410, 4)), 44100), **LIM)


def test_resampling_preserves_tone():
    fs_in, fs_out = 48000, 44100
    t = np.arange(fs_in) / fs_in
    x = np.sin(2 * np.pi * 1000 * t)
    y = resample(x, fs_in, fs_out)
    assert y.size == fs_out
    spec = np.abs(np.fft.rfft(y[2000:-2000] * np.hanning(y.size - 4000)))
    f = np.fft.rfftfreq(y.size - 4000, 1 / fs_out)
    assert abs(f[np.argmax(spec)] - 1000) < 2
    np.testing.assert_allclose(resample(x, 44100, 44100), x)


def test_stereo_policies(rng):
    l, r = rng.standard_normal(1000), rng.standard_normal(1000)
    x = np.stack([l, r], axis=1)
    mid = apply_stereo_policy(x, 44100, "mid")
    np.testing.assert_allclose(mid.signals[0], 0.5 * (l + r))
    assert 0.4 < mid.analysis["side_energy_fraction"] < 0.6
    assert apply_stereo_policy(x, 44100, "left").signals[0] is not l
    np.testing.assert_allclose(apply_stereo_policy(x, 44100, "right").signals[0], r)
    pair = apply_stereo_policy(x, 44100, "stereo_pair", width_deg=40)
    assert len(pair.signals) == 2 and pair.azimuth_offsets_deg == [20.0, -20.0]   # L to the left
    mono = apply_stereo_policy(x[:, :1], 44100, "stereo_pair")
    assert mono.policy == "mono" and len(mono.signals) == 1
    anti = analyse_stereo(np.stack([l, -l], axis=1))
    assert anti["warning"] and anti["correlation"] < -0.99
    with pytest.raises(AudioInputError):
        apply_stereo_policy(x, 44100, "surround")


def test_normalisation(rng):
    sustained = [0.01 * rng.standard_normal(44100)]
    (y,), g = normalize_sources(sustained, -20, -1)
    assert abs(20 * np.log10(np.sqrt(np.mean(y ** 2))) + 20) < 0.01
    transient = np.zeros(44100); transient[100] = 0.2
    (y,), _ = normalize_sources([transient], -20, -1)
    assert np.isclose(np.max(np.abs(y)), 10 ** (-1 / 20))           # capped by the peak ceiling
    (y,), g = normalize_sources([np.zeros(500)])
    assert g == 1.0 and not y.any()                                  # silence untouched


def test_short_clips_are_padded():
    assert ensure_min_length(np.ones(10), 128).shape == (128,)
    assert ensure_min_length(np.ones((10, 2)), 128).shape == (128, 2)


def test_limiter_never_exceeds_threshold(rng):
    lim = PeakLimiter(44100, 0.9, 50)
    x = rng.standard_normal((20000, 2)) * np.linspace(0.1, 4, 20000)[:, None]
    y = lim.process(x)
    assert np.max(np.abs(y)) <= 0.9 + 1e-12
    quiet = 0.1 * rng.standard_normal((1000, 2))
    np.testing.assert_array_equal(PeakLimiter(44100).process(quiet), quiet)   # transparent when idle


def test_wav_export_clips_safely():
    data = to_wav_bytes(np.array([[2.0, -3.0], [0.5, 0.1]]), 44100)
    y, _ = sf.read(io.BytesIO(data))
    assert np.max(np.abs(y)) <= 1.0


# ------------------------------------------------------------------ rendering

def test_static_render_matches_definition(interp, rng):
    x = rng.standard_normal(3000)
    h = interp.query_sph(60, 10).hrir
    for m in ("direct", "fft", "partitioned"):
        y = render_static(x, h, m)
        assert y.shape == (3000 + h.shape[1] - 1, 2)
        np.testing.assert_allclose(y[:, 0], direct_convolve(x, h[0]), atol=1e-9)
        np.testing.assert_allclose(y[:, 1], direct_convolve(x, h[1]), atol=1e-9)


def test_lateral_source_is_louder_and_earlier_in_near_ear(interp, rng):
    x = rng.standard_normal(8000)
    for az, near in ((90, 0), (270, 1)):
        y = render_static(x, interp.query_sph(az, 0).hrir)
        e = np.sum(y ** 2, axis=0)
        assert e[near] > 4 * e[1 - near]
        xc = np.correlate(y[:2000, 1 - near], y[:2000, near], "full")
        assert np.argmax(xc) - 1999 > 15            # far ear lags by > 0.35 ms


def test_silence_and_single_sample(interp):
    h = interp.query_sph(0, 0).hrir
    assert not render_static(np.zeros(500), h).any()
    y = render_static(np.array([1.0]), h)
    np.testing.assert_allclose(y.T, h)
    assert render_static(np.zeros(0), h).shape == (0, 2)


def test_block_engine_static_equals_reference(interp, rng):
    x = 0.1 * rng.standard_normal(5000)
    pose = ListenerPose(np.zeros(3))
    src = np.array([-0.8, 0.4, -1.1])
    from sonicsphere.coordinate_system import world_to_head
    d = world_to_head(src, np.zeros(3))
    r = float(np.linalg.norm(d))
    for method in ("direct", "partitioned"):
        y, eng = render_moving([x], [0.0], interp, lambda f: src, pose, gain=1.0, method=method,
                               distance_attenuation=False)
        assert eng.stats.filter_updates == 1 and eng.stats.crossfades == 0
        ref = render_static(x, interp.query_quantized(d / r).hrir)
        np.testing.assert_allclose(y, ref, atol=1e-9)


def test_crossfade_removes_switching_clicks(interp):
    """A sine through an abruptly switched filter has a large sample-to-sample jump at the
    switch; the per-block crossfade keeps the output as smooth as a static render."""
    fs = interp.pf.fs
    n = 128 * 40
    x = 0.5 * np.sin(2 * np.pi * 300 * np.arange(n) / fs)
    pose = ListenerPose(np.zeros(3))
    a, b = sph_to_cart(80, 0), sph_to_cart(280, 0)          # left -> right: large filter change
    pos = lambda f: a if f < 128 * 20 else b
    smooth, eng = render_moving([x], [0.0], interp, pos, pose, crossfade=True, distance_attenuation=False, tail=False)
    hard, _ = render_moving([x], [0.0], interp, pos, pose, crossfade=False, distance_attenuation=False, tail=False)
    assert eng.stats.crossfades == 1
    sw = 128 * 20
    jump = lambda y: np.max(np.abs(np.diff(y[sw - 4: sw + 4], axis=0)))
    typical = np.max(np.abs(np.diff(smooth[1000:2000], axis=0)))
    assert jump(hard) > 3 * typical
    assert jump(smooth) <= 1.5 * typical


def test_trajectory_render_follows_path(interp, rng):
    """A source circling the head: left-ear energy peaks when the source is on the left."""
    fs = interp.pf.fs
    pts = [sph_to_cart(a, 0) * 1.5 for a in range(0, 360, 45)]
    world = [np.array([-p[1], p[2], -p[0]]) for p in pts]       # head frame -> world at yaw 0
    traj = Trajectory.from_points(np.array(world), duration_s=2.0, loop=True)
    x = 0.1 * rng.standard_normal(int(2.0 * fs))
    from sonicsphere.audio_renderer import render_trajectory
    y, eng = render_trajectory([x], [0.0], interp, traj, ListenerPose(np.zeros(3)), tail=False,
                               distance_attenuation=False)
    assert eng.stats.filter_updates > 300 and eng.stats.crossfades == eng.stats.filter_updates - 1
    frames = y.shape[0] // 8
    ild = [10 * np.log10(np.sum(y[i * frames:(i + 1) * frames, 0] ** 2) / np.sum(y[i * frames:(i + 1) * frames, 1] ** 2))
           for i in range(8)]
    # Path starts at the front (az 0) and moves counter-clockwise: left side around 1/4 of the loop.
    assert np.argmax(ild) in (1, 2) and np.argmin(ild) in (5, 6)
    assert max(ild) > 6 and min(ild) < -6


def test_stereo_pair_render(interp, rng):
    x = 0.1 * rng.standard_normal(4000)
    pair = apply_stereo_policy(np.stack([x, np.zeros_like(x)], axis=1), 44100, "stereo_pair", 120)
    y, _ = render_moving(pair.signals, pair.azimuth_offsets_deg, interp, lambda f: np.array([0, 0, -1.0]),
                         ListenerPose(np.zeros(3)), distance_attenuation=False)
    e = np.sum(y ** 2, axis=0)
    assert e[0] > 2 * e[1]       # only the left channel has signal and it sits 60 deg to the left


def test_distance_gain():
    assert distance_gain(1.0) == 1.0 and distance_gain(2.0) == 0.5
    assert distance_gain(0.01) == 2.0 and distance_gain(100) == 0.25
    assert distance_gain(3.0, enabled=False) == 1.0


def test_transient_and_sustained_inputs(interp):
    fs = interp.pf.fs
    pose = ListenerPose(np.zeros(3))
    clicks = np.zeros(int(0.5 * fs)); clicks[::4410] = 0.9
    tone = 0.3 * np.sin(2 * np.pi * 220 * np.arange(int(0.5 * fs)) / fs)
    for x in (clicks, tone):
        y, eng = render_moving([x], [0.0], interp, lambda f: np.array([1.0, 0.2, -1.0]), pose, gain=interp.pf.gain)
        assert np.all(np.isfinite(y)) and np.max(np.abs(y)) <= 0.98 + 1e-12
        assert np.max(np.abs(y)) > 0.01
