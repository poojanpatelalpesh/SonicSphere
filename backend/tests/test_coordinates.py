import numpy as np
import pytest

from sonicsphere.coordinate_system import (angular_distance_deg, cart_to_interaural_polar, cart_to_sph,
                                           interaural_polar_to_cart, listener_basis, quantize_direction,
                                           rotate_about_up, sph_to_cart, world_to_head, world_to_sofa_direction)


def test_sofa_cardinal_directions():
    np.testing.assert_allclose(sph_to_cart(0, 0), [1, 0, 0], atol=1e-12)    # front
    np.testing.assert_allclose(sph_to_cart(90, 0), [0, 1, 0], atol=1e-12)   # left
    np.testing.assert_allclose(sph_to_cart(270, 0), [0, -1, 0], atol=1e-12)  # right
    np.testing.assert_allclose(sph_to_cart(0, 90), [0, 0, 1], atol=1e-12)   # up


def test_sph_roundtrip_and_wrap(rng):
    az = rng.uniform(0, 360, 500)
    el = rng.uniform(-89, 89, 500)
    a2, e2, r2 = cart_to_sph(sph_to_cart(az, el, 2.0))
    np.testing.assert_allclose(np.cos(np.deg2rad(a2 - az)), 1, atol=1e-12)
    np.testing.assert_allclose(e2, el, atol=1e-10)
    np.testing.assert_allclose(r2, 2.0)
    a, _, _ = cart_to_sph(np.array([1.0, -1e-17, 0.0]))
    assert 0 <= a < 360
    a, _, _ = cart_to_sph(sph_to_cart(-30, 0))
    assert np.isclose(a, 330)


def test_interaural_polar_roundtrip(rng):
    lat = rng.uniform(-89, 89, 200)
    pol = rng.uniform(-90, 269, 200)
    l2, p2 = cart_to_interaural_polar(interaural_polar_to_cart(lat, pol))
    np.testing.assert_allclose(l2, lat, atol=1e-9)
    np.testing.assert_allclose(p2, pol, atol=1e-9)


def test_listener_basis_is_orthonormal_right_handed():
    for yaw in (0, 37, -120, 180):
        for pitch in (0, 20, -45):
            b = listener_basis(yaw, pitch)
            np.testing.assert_allclose(b @ b.T, np.eye(3), atol=1e-12)
            np.testing.assert_allclose(np.cross(b[0], b[1]), b[2], atol=1e-12)  # fwd x left = up


def test_world_to_head_default_pose():
    pl = np.array([0, 1.2, 0])
    # Listener faces -Z: a source at -Z is in front, -X is on the left, +Y above.
    for p, (az, el) in [((0, 1.2, -2), (0, 0)), ((-2, 1.2, 0), (90, 0)), ((2, 1.2, 0), (270, 0)),
                        ((0, 1.2, 2), (180, 0)), ((0, 3.2, 0), (None, 90))]:
        a, e, r = world_to_sofa_direction(np.array(p, float), pl)
        assert np.isclose(r, 2.0)
        assert np.isclose(e, el, atol=1e-9)
        if az is not None:
            assert np.isclose(np.cos(np.deg2rad(a - az)), 1, atol=1e-12)


def test_turning_the_listener_changes_direction():
    pl = np.zeros(3)
    src = np.array([-1.0, 0.0, 0.0])   # world -X: on the left at yaw 0
    a0, _, _ = world_to_sofa_direction(src, pl, yaw_deg=0)
    a1, _, _ = world_to_sofa_direction(src, pl, yaw_deg=90)   # turn left to face it
    a2, _, _ = world_to_sofa_direction(src, pl, yaw_deg=-90)  # turn right: it is now behind
    assert np.isclose(a0, 90)
    assert np.isclose(np.cos(np.deg2rad(a1)), 1.0, atol=1e-12)   # now straight ahead
    assert np.isclose(a2, 180)
    # Pitching the head up moves a frontal source downwards in head coordinates.
    _, e, _ = world_to_sofa_direction(np.array([0, 0, -1.0]), pl, pitch_deg=30)
    assert np.isclose(e, -30)


def test_head_world_roundtrip(rng):
    from sonicsphere.coordinate_system import head_to_world
    pl = np.array([0.3, 1.1, -0.2])
    p = rng.standard_normal((50, 3))
    h = world_to_head(p, pl, 33, -12)
    np.testing.assert_allclose(head_to_world(h, pl, 33, -12), p, atol=1e-12)


def test_angular_distance_and_rotation():
    assert np.isclose(angular_distance_deg(sph_to_cart(10, 0), sph_to_cart(350, 0)), 20)
    assert np.isclose(angular_distance_deg(sph_to_cart(0, 89.9), sph_to_cart(180, 89.9)), 0.2, atol=1e-6)
    v = rotate_about_up(sph_to_cart(0, 0), 30)
    assert np.isclose(cart_to_sph(v)[0], 30)


@pytest.mark.parametrize("az,el,exp", [(359.96, 10, (0.0, 10.0)), (12.34, -45.06, (12.3, -45.1)),
                                       (100, 89.97, (0.0, 90.0)), (0.05, 0.05, (0.1, 0.1))])
def test_quantize_direction(az, el, exp):
    a, e = quantize_direction(az, el, 0.1)
    assert np.isclose(a, exp[0]) and np.isclose(e, exp[1])
