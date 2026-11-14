"""Coordinate systems used by SonicSphere.

World frame (identical to the Three.js scene)
    Right-handed, metres, +Y up. With the listener at yaw = 0 the listener looks
    along -Z, so +X is to the listener's right and -X to the left.

Listener pose
    Position ``p_l`` (centre of the head, midway between the ears), yaw ``psi``
    (rotation about +Y, positive = turn to the listener's LEFT, i.e. counter-clockwise
    seen from above) and pitch ``theta`` (positive = nose up).

Head frame (SOFA "SimpleFreeFieldHRIR" convention, used by the dataset)
    x = forward (out of the nose), y = left (through the left ear), z = up.

Spherical (SOFA)
    azimuth in [0, 360): 0 = front, 90 = left, 180 = back, 270 = right
    (counter-clockwise seen from above); elevation in [-90, 90]: 0 = horizontal
    plane, +90 = above the head.

Interaural-polar (CIPIC native)
    lateral angle  = asin(y) in [-90, 90], positive to the left
    polar angle    = atan2(z, x) in [-90, 270): 0 = front, 90 = above, 180 = behind
"""

from __future__ import annotations

import math

import numpy as np

ArrayLike = np.ndarray | float


def _wrap360(az: np.ndarray) -> np.ndarray:
    az = np.mod(az, 360.0)
    return np.where(az >= 360.0, az - 360.0, az)


def sph_to_cart(az_deg: ArrayLike, el_deg: ArrayLike, r: ArrayLike = 1.0) -> np.ndarray:
    """SOFA spherical (deg, deg, m) -> head-frame cartesian (..., 3)."""
    az = np.deg2rad(np.asarray(az_deg, dtype=float))
    el = np.deg2rad(np.asarray(el_deg, dtype=float))
    r = np.asarray(r, dtype=float)
    return np.stack([r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el)], axis=-1)


def cart_to_sph(v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Head-frame cartesian (..., 3) -> (azimuth [0,360), elevation [-90,90], radius)."""
    v = np.asarray(v, dtype=float)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    r = np.sqrt(x * x + y * y + z * z)
    az = _wrap360(np.rad2deg(np.arctan2(y, x)))
    el = np.rad2deg(np.arctan2(z, np.hypot(x, y)))
    return az, el, r


def cart_to_interaural_polar(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Head-frame cartesian -> (lateral, polar) in degrees (CIPIC-style)."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1)
    n = np.where(n == 0, 1.0, n)
    x, y, z = v[..., 0] / n, v[..., 1] / n, v[..., 2] / n
    lateral = np.rad2deg(np.arcsin(np.clip(y, -1.0, 1.0)))
    polar = np.rad2deg(np.arctan2(z, x))
    polar = np.where(polar < -90.0, polar + 360.0, polar)
    return lateral, polar


def interaural_polar_to_cart(lateral_deg: ArrayLike, polar_deg: ArrayLike) -> np.ndarray:
    lat = np.deg2rad(np.asarray(lateral_deg, dtype=float))
    pol = np.deg2rad(np.asarray(polar_deg, dtype=float))
    return np.stack([np.cos(lat) * np.cos(pol), np.sin(lat), np.cos(lat) * np.sin(pol)], axis=-1)


def listener_basis(yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> np.ndarray:
    """Rows = listener forward, left and up axes expressed in world coordinates."""
    psi = np.deg2rad(yaw_deg)
    th = np.deg2rad(pitch_deg)
    forward = np.array([-np.sin(psi) * np.cos(th), np.sin(th), -np.cos(psi) * np.cos(th)])
    left = np.array([-np.cos(psi), 0.0, np.sin(psi)])
    up = np.cross(forward, left)
    return np.stack([forward, left, up])


def world_to_head(p_source: np.ndarray, p_listener: np.ndarray, yaw_deg: float = 0.0,
                  pitch_deg: float = 0.0) -> np.ndarray:
    """d(t) = p_s(t) - p_l expressed in the head frame (x fwd, y left, z up)."""
    d = np.asarray(p_source, dtype=float) - np.asarray(p_listener, dtype=float)
    return d @ listener_basis(yaw_deg, pitch_deg).T


def head_to_world(v_head: np.ndarray, p_listener: np.ndarray, yaw_deg: float = 0.0,
                  pitch_deg: float = 0.0) -> np.ndarray:
    return np.asarray(v_head, dtype=float) @ listener_basis(yaw_deg, pitch_deg) + np.asarray(p_listener)


def world_to_sofa_direction(p_source: np.ndarray, p_listener: np.ndarray, yaw_deg: float = 0.0,
                            pitch_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World-space source position -> SOFA (azimuth, elevation, distance)."""
    return cart_to_sph(world_to_head(p_source, p_listener, yaw_deg, pitch_deg))


def angular_distance_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Great-circle angle between direction vectors (broadcasting over leading axes)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = a / np.linalg.norm(a, axis=-1, keepdims=True)
    nb = b / np.linalg.norm(b, axis=-1, keepdims=True)
    # atan2 formulation is accurate for both tiny and near-180 degree angles.
    cross = np.linalg.norm(np.cross(na, nb), axis=-1)
    dot = np.sum(na * nb, axis=-1)
    return np.rad2deg(np.arctan2(cross, dot))


def rotate_about_up(v_head: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate head-frame vectors about the head's z (up) axis; positive = towards the left."""
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    v = np.asarray(v_head, dtype=float)
    return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1], v[..., 2]], axis=-1)


def quantize_direction(az_deg: float, el_deg: float, quantum_deg: float) -> tuple[float, float]:
    """Quantise a direction for filter caching. Must match frontend/src/dsp/coordinates.ts."""
    # floor(x + 0.5) (round half up) is used instead of numpy's round-half-even so that
    # the browser (Math.floor) produces bit-identical keys.
    q = quantum_deg
    el_k = math.floor(el_deg / q + 0.5)
    az_k = math.floor(az_deg / q + 0.5)
    el_q = min(90.0, max(-90.0, el_k * q))
    az_q = az_k * q
    if az_q >= 360.0 - 1e-9:
        az_q -= 360.0
    if abs(el_q) >= 90.0 - 1e-9:  # azimuth is meaningless at the poles
        az_q = 0.0
    return az_q, el_q
