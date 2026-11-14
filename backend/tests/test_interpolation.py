import numpy as np
import pytest

from sonicsphere.coordinate_system import angular_distance_deg, cart_to_sph, sph_to_cart
from sonicsphere.hrir_interpolation import (METHODS, DirectionWeighter, HRIRInterpolator, build_triangulation,
                                            locate)
from sonicsphere.hrir_preprocessing import (estimate_toa, frac_shift, itd_cross_correlation, minimum_phase,
                                            reconstruct)
from sonicsphere.interpolation_eval import _subset, evaluate_subject, lsd_db, nmse_db


def _random_dirs(rng, n):
    v = rng.standard_normal((n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ---------------------------------------------------------------- preprocessing

def test_frac_shift_integer_and_inverse(rng):
    h = np.zeros(64); h[5] = 1.0
    y = frac_shift(h, 3, 128)
    assert np.argmax(y) == 8 and np.isclose(y[8], 1.0, atol=1e-9)
    x = rng.standard_normal(100) * np.hanning(100)
    for s in (0.37, -2.6, 11.25):
        back = frac_shift(frac_shift(x, s, 256), -s, 256)[:100]
        assert np.max(np.abs(back - x)) < 0.02 * np.max(np.abs(x))  # only the Nyquist bin differs


def test_toa_on_synthetic_delay():
    from scipy.signal import firwin
    base = firwin(31, 0.8)        # short band-limited pulse
    for d in (10.0, 10.25, 17.5, 30.8):
        h = frac_shift(np.pad(base, (0, 97)), d, 256)[:128]
        h0 = np.pad(base, (0, 97))
        assert abs((estimate_toa(h) - estimate_toa(h0)) - d) < 0.1


def test_toa_matches_cross_correlation_itd(prepared):
    pf = prepared.filters
    # Lateral sources: TOA-based ITD agrees with the interaural cross-correlation ITD.
    idx = [i for i in range(len(pf.vectors)) if abs(pf.vectors[i, 1]) > 0.5][::10]
    d = [(pf.toa[i, 1] - pf.toa[i, 0]) / pf.fs - itd_cross_correlation(pf.measured[i, 0], pf.measured[i, 1], pf.fs)
         for i in idx]
    assert np.median(np.abs(d)) < 40e-6
    left = int(np.argmax(pf.vectors @ np.array([0, 1.0, 0])))
    assert pf.toa[left, 1] - pf.toa[left, 0] > 20       # ~0.6 ms: right ear lags for a left source


def test_alignment_roundtrip_is_accurate(prepared):
    pf = prepared.filters
    rec = reconstruct(pf.aligned, pf.toa, pf.margin, pf.filter_length)
    e = nmse_db(rec, pf.measured)
    assert np.median(e) < -60 and np.percentile(e, 95) < -40 and e.max() < -35
    # Every aligned shape crosses its onset threshold at the margin.
    t = estimate_toa(pf.aligned)
    assert np.median(np.abs(t - pf.margin)) < 0.05


def test_minimum_phase_keeps_magnitude(rng):
    h = rng.standard_normal(128) * np.exp(-np.arange(128) / 20)
    m = minimum_phase(h, 2048)
    H = np.abs(np.fft.rfft(h, 2048)); M = np.abs(np.fft.rfft(m, 2048))
    assert np.median(np.abs(20 * np.log10(M / H))) < 0.5
    assert np.sum(m[:10] ** 2) >= np.sum(h[:10] ** 2)   # energy concentrated at the start


# ---------------------------------------------------------------- triangulation / weights

def test_triangulation_covers_sphere(prepared, rng):
    tri = prepared.triangulation
    for p in _random_dirs(rng, 400):
        t, w = locate(tri, p)
        assert w.min() >= -1e-9


def _ip(az, el):
    from sonicsphere.coordinate_system import cart_to_interaural_polar
    return cart_to_interaural_polar(sph_to_cart(az, el))


def test_cipic_gap_is_the_unmeasured_polar_wedge(prepared):
    """CIPIC measures polar angles -45..230.625 deg; the gap triangles are exactly the wedge
    in between (below the head), at all lateral angles."""
    from sonicsphere.coordinate_system import cart_to_interaural_polar
    tri = prepared.triangulation
    assert tri.is_gap.sum() > 0 and tri.boundary_edges.shape[0] > 0
    c = tri.vertices[tri.triangles[tri.is_gap]].mean(axis=1)
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    _, pol = cart_to_interaural_polar(c)
    assert np.all(c[:, 2] < -0.25)                                   # below the horizon
    assert np.all((pol > 225) | (pol < -40))                         # inside the unmeasured wedge


def test_weights_valid_and_measured_exact(interp, rng):
    pf = interp.pf
    for p in _random_dirs(rng, 300):
        r = interp.query(p)
        assert np.isclose(r.w.weights.sum(), 1.0) and np.all(r.w.weights >= 0)
        assert len(r.w.indices) <= (3 if r.w.status == "interpolated" else 4)
    for i in (0, 100, 608, 1249):
        r = interp.query(pf.vectors[i])
        assert r.w.status == "measured" and r.w.indices[0] == i
        np.testing.assert_array_equal(r.hrir, pf.measured[i])


def test_status_regions(interp):
    from sonicsphere.coordinate_system import interaural_polar_to_cart
    for lat in range(-75, 80, 15):
        for pol in (-35.3, 0.7, 47.1, 90.4, 149.0, 220.2):
            assert interp.query(interaural_polar_to_cart(lat, pol)).w.status == "interpolated", (lat, pol)
        if abs(lat) <= 60:   # at larger |lateral| the wedge is narrower than the gap threshold
            for pol in (240.0, 270.0, 300.0):
                assert interp.query(interaural_polar_to_cart(lat, pol)).w.status == "gap_fill", (lat, pol)
    assert interp.query_sph(0, -90).w.status == "gap_fill"           # straight down


def test_gap_fill_follows_cone_of_confusion(interp):
    """Inside the gap the weights move linearly in polar angle between the two rim points
    of the same lateral circle, and are continuous with the measured region at the rim."""
    from sonicsphere.coordinate_system import interaural_polar_to_cart
    lat = 30.0
    r = interp.query(interaural_polar_to_cart(lat, 270.0)).w
    assert r.status == "gap_fill"
    assert abs(r.extra["lateral_deg"] - lat) < 1e-9
    assert abs(r.extra["polar_hi_deg"] % 360 - 315.0) < 0.5 and abs(r.extra["polar_lo_deg"] % 360 - 230.625) < 0.5
    assert 0.4 < r.extra["t"] < 0.6 and len(r.indices) <= 4
    inside = interp.query(interaural_polar_to_cart(lat, 230.5)).hrir      # just inside the gap
    rim = interp.query(interaural_polar_to_cart(lat, 230.7)).hrir         # just inside the data
    assert nmse_db(inside, rim).max() < -25
    # ITD stays that of the lateral angle across the whole gap.
    itds = [np.diff(interp.query(interaural_polar_to_cart(lat, b)).toa)[0] for b in np.arange(232, 314, 4)]
    assert np.ptp(itds) < 3.0                                              # < 70 us variation


def test_clamp_fallback_near_interaural_axis(prepared):
    from sonicsphere.hrir_interpolation import clamp_to_boundary
    tri = prepared.triangulation
    q, i, j, alpha = clamp_to_boundary(tri, sph_to_cart(30, -80))
    _, el, _ = cart_to_sph(q)
    assert -60 < el < -20 and 0 <= alpha <= 1


def test_azimuth_wraparound_continuity(interp):
    """Crossing az = 0/360 behaves like any other step."""
    steps = np.arange(-6, 6.01, 0.25)
    filters = [interp.query_sph(a % 360, 10.0).hrir for a in steps]
    d = [float(np.sum((b - a) ** 2) / np.sum(a ** 2)) for a, b in zip(filters, filters[1:])]
    mid = len(d) // 2
    assert max(d[mid - 2: mid + 2]) <= 3 * np.median(d) + 1e-6
    a, b = interp.query_sph(359.95, 10.0), interp.query_sph(0.05, 10.0)
    assert nmse_db(a.hrir, b.hrir).max() < -25


def test_smooth_along_paths(interp):
    """Small direction steps give small filter changes everywhere in the measured region."""
    for el in (-60.0, -20.0, 0.0, 35.0, 70.0):
        prev = None
        worst = -np.inf
        for az in np.arange(0, 360, 0.5):
            h = interp.query_sph(az, el).hrir
            if prev is not None:
                worst = max(worst, float(nmse_db(h, prev).max()))
            prev = h
        assert worst < -8, (el, worst)


def test_converges_to_measurement(interp):
    """Just outside the snap radius the interpolated filter matches the measured one."""
    from sonicsphere.coordinate_system import cart_to_interaural_polar
    pf = interp.pf
    lat, pol = cart_to_interaural_polar(pf.vectors)
    inner = np.nonzero((np.abs(lat) < 60) & (pol > 0) & (pol < 180))[0]
    for i in inner[::97]:
        az, el, _ = cart_to_sph(pf.vectors[i])
        near = interp.query_sph(float(az), float(el) + 0.3)
        assert near.w.status == "interpolated"
        assert nmse_db(near.hrir, pf.measured[i]).max() < -15


def test_interaural_timing_is_interpolated(interp):
    """TOAs are convex combinations; the rendered filter's onset matches the interpolated TOA."""
    pf = interp.pf
    for az, el in ((37, 12), (101, -20), (250, 45), (170, 5)):
        r = interp.query_sph(az, el)
        lo, hi = pf.toa[r.w.indices].min(axis=0), pf.toa[r.w.indices].max(axis=0)
        assert np.all(r.toa >= lo - 1e-9) and np.all(r.toa <= hi + 1e-9)
        est = estimate_toa(r.hrir)
        assert np.all(np.abs(est - r.toa) < 0.5)                   # < 11 us at 44.1 kHz
        itd = (est[1] - est[0]) - (r.toa[1] - r.toa[0])
        assert abs(itd) / pf.fs < 10e-6


def test_idw_and_nearest(prepared, rng):
    pf = prepared.filters
    near = HRIRInterpolator(pf, "nearest")
    idw = HRIRInterpolator(pf, "idw_aligned")
    for p in _random_dirs(rng, 50):
        r = near.query(p)
        assert r.w.status == "measured"
        assert angular_distance_deg(pf.vectors[r.w.indices[0]], p) <= angular_distance_deg(pf.vectors, p).min() + 1e-9
        w = idw.query(p).w
        assert len(w.indices) == 4 and np.isclose(w.weights.sum(), 1)
    np.testing.assert_array_equal(idw.query(pf.vectors[5]).hrir, pf.measured[5])


def test_missing_measurements_are_handled(prepared):
    """Remove a whole region of measurements: the hole is detected and clamped to its rim."""
    pf = prepared.filters
    _, el, _ = cart_to_sph(pf.vectors)
    az, _, _ = cart_to_sph(pf.vectors)
    hole = (np.abs(((az - 90 + 180) % 360) - 180) < 25) & (np.abs(el - 10) < 25)
    sub = _subset(pf, np.nonzero(~hole)[0])
    it = HRIRInterpolator(sub, "barycentric_aligned", gap_edge_threshold_deg=30.0)
    assert hole.sum() > 20
    assert it.triangulation.is_gap.sum() > prepared.triangulation.is_gap.sum()  # the new hole is detected
    r = it.query_sph(90, 10)
    assert r.w.status in ("gap_fill", "clamped") and np.isclose(r.w.weights.sum(), 1)
    assert np.all(np.isfinite(r.hrir))


def test_rejects_unknown_method(prepared):
    with pytest.raises(ValueError):
        HRIRInterpolator(prepared.filters, "cubic_magic")
    with pytest.raises(ValueError):
        DirectionWeighter(prepared.filters.vectors, "bogus")


def test_degenerate_point_sets():
    with pytest.raises(ValueError):
        build_triangulation(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 0, 0.0]]))


# ---------------------------------------------------------------- held-out experiment

def test_holdout_selected_method_beats_baselines(prepared, settings):
    res = evaluate_subject(prepared.filters, 0.1, 7, settings,
                           methods=["nearest", "barycentric_raw", "barycentric_aligned"])
    m = res["methods"]
    assert res["held_out"] >= 100
    assert m["barycentric_aligned"]["lsd_db"] < m["nearest"]["lsd_db"]
    assert m["barycentric_aligned"]["lsd_db"] < m["barycentric_raw"]["lsd_db"]
    assert m["barycentric_aligned"]["itd_err_us"] < m["nearest"]["itd_err_us"]
    assert m["barycentric_aligned"]["nmse_db"] < m["nearest"]["nmse_db"]
    assert m["barycentric_aligned"]["lsd_db"] < 3.0 and m["barycentric_aligned"]["itd_err_us"] < 20


def test_lsd_metric_sanity(rng):
    h = rng.standard_normal((3, 128))
    assert np.allclose(lsd_db(h, h, 44100), 0)
    assert np.allclose(lsd_db(2 * h, h, 44100), 20 * np.log10(2))


def test_all_methods_produce_finite_filters(prepared, rng):
    for m in METHODS:
        it = HRIRInterpolator(prepared.filters, m)
        for p in _random_dirs(rng, 10):
            assert np.all(np.isfinite(it.query(p).hrir))
