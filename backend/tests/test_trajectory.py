import numpy as np
import pytest

from sonicsphere.coordinate_system import world_to_sofa_direction
from sonicsphere.trajectory import Trajectory, resample_by_arclength, spline_polyline


def square(r=1.0):
    return np.array([[r, 0, 0], [0, 0, -r], [-r, 0, 0], [0, 0, r]], float)


def test_spline_passes_through_control_points():
    p = square()
    poly = spline_polyline(p, closed=True)
    for q in p:
        assert np.min(np.linalg.norm(poly - q, axis=1)) < 1e-9
    open_poly = spline_polyline(p, closed=False)
    np.testing.assert_allclose(open_poly[0], p[0]); np.testing.assert_allclose(open_poly[-1], p[-1])


def test_uniform_speed_and_smoothness():
    tr = Trajectory.from_points(square(1.5), duration_s=4.0, loop=True)
    seg = np.linalg.norm(np.diff(tr.samples, axis=0), axis=1)
    assert seg.max() / seg.min() < 1.05                    # arc-length uniform
    d1 = np.diff(tr.samples, axis=0)
    cosang = np.sum(d1[1:] * d1[:-1], axis=1) / (np.linalg.norm(d1[1:], axis=1) * np.linalg.norm(d1[:-1], axis=1))
    assert cosang.min() > np.cos(np.deg2rad(8))            # no kinks (C1-continuous curve)
    np.testing.assert_allclose(tr.samples[0], tr.samples[-1], atol=1e-9)   # closed loop


def test_looping_and_clamping():
    tr = Trajectory.from_points(square(), duration_s=2.0, loop=True)
    np.testing.assert_allclose(tr.position_at(0.5), tr.position_at(2.5), atol=1e-12)
    np.testing.assert_allclose(tr.position_at(-1.5), tr.position_at(0.5), atol=1e-12)
    once = Trajectory.from_points(square(), duration_s=2.0, loop=False)
    np.testing.assert_allclose(once.position_at(5.0), once.samples[-1])
    np.testing.assert_allclose(once.position_at(-1.0), once.samples[0])


def test_listener_relative_direction_along_trajectory():
    """Circle around a listener at (0, 1.2, 0): azimuth increases monotonically (CCW)."""
    pl = np.array([0, 1.2, 0])
    ang = np.deg2rad(np.arange(0, 360, 30))
    pts = np.stack([-np.sin(ang), np.full_like(ang, 1.2), -np.cos(ang)], axis=1) * [1.5, 1, 1.5]
    pts[:, 1] = 1.2
    tr = Trajectory.from_points(pts, duration_s=12.0)
    az = [float(world_to_sofa_direction(tr.position_at(t), pl)[0]) for t in np.arange(0, 12, 0.25)]
    unwrapped = np.unwrap(np.deg2rad(az))
    assert np.all(np.diff(unwrapped) > 0)
    assert abs(np.rad2deg(unwrapped[-1] - unwrapped[0]) - 352.5) < 3


def test_degenerate_inputs():
    one = Trajectory.from_points(np.array([[1.0, 2, 3]]), 1.0)
    np.testing.assert_allclose(one.position_at(0.3), [1, 2, 3])
    two = Trajectory.from_points(np.array([[0, 0, 0], [2.0, 0, 0]]), 1.0, loop=False)
    np.testing.assert_allclose(two.position_at(0.5), [1, 0, 0], atol=1e-9)
    same = resample_by_arclength(np.zeros((5, 3)), 8)
    assert same.shape == (8, 3) and not same.any()
    with pytest.raises(ValueError):
        Trajectory.from_points(np.zeros((0, 3)), 1.0)
    with pytest.raises(ValueError):
        Trajectory.from_points(square(), 0.0)
