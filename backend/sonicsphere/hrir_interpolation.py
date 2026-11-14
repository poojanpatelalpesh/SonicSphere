"""Spatial interpolation of HRIRs between measured directions.

Weighting schemes
    * nearest      - the closest measured direction (great-circle distance)
    * idw          - inverse-distance weighting of the k nearest measurements
    * barycentric  - spherical barycentric weights inside the triangle of a spherical
                     Delaunay triangulation (convex hull of the unit direction vectors);
                     weights are continuous across triangle edges.

Representations the weights are applied to
    * raw          - the measured HRIR samples
    * aligned      - onset-aligned shapes plus separately interpolated TOAs
                     (the onset delay is re-inserted with a fractional shift)
    * minphase     - minimum-phase shapes plus interpolated TOAs

Unmeasured regions
    Hull triangles with an edge longer than ``gap_edge_threshold_deg`` span a region
    without measurements. For CIPIC this is the wedge of polar angles 230.6..315 degrees
    (below and behind/in front of the head) at every lateral angle.

    Gap fill along cones of confusion ("gap_fill"): a direction p in the gap has lateral
    angle a = asin(p_y) and polar angle b. The circle of constant lateral angle through p
    (a cone of confusion; ITD and ILD depend mainly on a) crosses the boundary of the
    measured region at polar angles b_lo < b < b_hi. The HRIR is interpolated linearly in
    polar angle between the two boundary points, each of which lies on a boundary edge
    and is itself a combination of that edge's two vertices. Weights are continuous at
    the rim (they equal the barycentric weights there) and inside the gap.

    Fallback ("clamped"): if the lateral circle does not cross the boundary twice (only
    possible at |a| ~ 90 degrees), the direction is clamped to the closest boundary point.

The browser implementation (frontend/src/dsp/interpolation.ts) mirrors this module;
``export_golden.py`` produces test vectors that keep the two in agreement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.spatial import ConvexHull

from .coordinate_system import angular_distance_deg, cart_to_sph, quantize_direction, sph_to_cart
from .hrir_preprocessing import PreparedFilters, reconstruct

INSIDE_EPS = 1e-9


@dataclass
class Triangulation:
    vertices: np.ndarray        # (M, 3) unit vectors
    triangles: np.ndarray       # (T, 3) int, counter-clockwise seen from outside
    inverse: np.ndarray         # (T, 3, 3) inverse of [v0 v1 v2] (vertices as columns)
    max_edge_deg: np.ndarray    # (T,)
    is_gap: np.ndarray          # (T,) bool
    boundary_edges: np.ndarray  # (E, 2) vertex index pairs on the edge of the measured region

    @property
    def num_triangles(self) -> int:
        return int(self.triangles.shape[0])


def build_triangulation(vectors: np.ndarray, gap_edge_threshold_deg: float = 30.0) -> Triangulation:
    v = np.asarray(vectors, dtype=float)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    # Duplicate directions would create degenerate facets.
    _, uniq = np.unique(np.round(v, 9), axis=0, return_index=True)
    if len(uniq) != len(v):
        raise ValueError("Duplicate measurement directions; deduplicate before triangulating")
    hull = ConvexHull(v)
    tri = hull.simplices.astype(np.int64).copy()
    # Orient every facet counter-clockwise seen from outside.
    a, b, c = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
    flip = np.einsum("ij,ij->i", np.cross(b - a, c - a), a + b + c) < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    # The hull of points on a sphere always contains the origin only if the points are
    # spread over more than a hemisphere; verify, otherwise barycentric weights break.
    if np.any(hull.equations[:, 3] > -1e-9):
        raise ValueError("Measurement directions do not enclose the listener; cannot triangulate")
    mats = np.stack([v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]], axis=2)  # columns = vertices
    inverse = np.linalg.inv(mats)
    edges = np.stack([
        angular_distance_deg(v[tri[:, 0]], v[tri[:, 1]]),
        angular_distance_deg(v[tri[:, 1]], v[tri[:, 2]]),
        angular_distance_deg(v[tri[:, 2]], v[tri[:, 0]]),
    ], axis=1)
    max_edge = edges.max(axis=1)
    is_gap = max_edge > gap_edge_threshold_deg

    # Boundary edges: shared by exactly one valid and one gap triangle.
    owners: dict[tuple[int, int], list[int]] = {}
    for t, (i, j, k) in enumerate(tri):
        for e in ((i, j), (j, k), (k, i)):
            owners.setdefault((min(e), max(e)), []).append(t)
    boundary = [e for e, ts in owners.items() if len(ts) == 2 and is_gap[ts[0]] != is_gap[ts[1]]]
    boundary_edges = np.array(sorted(boundary), dtype=np.int64).reshape(-1, 2)
    return Triangulation(vertices=v, triangles=tri, inverse=inverse, max_edge_deg=max_edge,
                         is_gap=is_gap, boundary_edges=boundary_edges)


def locate(tr: Triangulation, p: np.ndarray) -> tuple[int, np.ndarray]:
    """Triangle containing direction p and its (unnormalised) barycentric coordinates.

    The triangle maximising the smallest barycentric coordinate is selected; for a
    point inside the triangulation exactly one triangle has all coordinates >= 0.
    """
    w = tr.inverse @ p                       # (T, 3)
    t = int(np.argmax(w.min(axis=1)))
    return t, w[t]


def clamp_to_boundary(tr: Triangulation, p: np.ndarray) -> tuple[np.ndarray, int, int, float]:
    """Closest point to p on the great-circle arcs bounding the measured region.

    Returns (q, i, j, alpha): the clamped unit vector and its position as the
    normalised combination q ~ (1-alpha) v_i + alpha v_j of the edge endpoints.
    """
    if tr.boundary_edges.size == 0:
        raise ValueError("Triangulation has no boundary edges")
    a = tr.vertices[tr.boundary_edges[:, 0]]
    b = tr.vertices[tr.boundary_edges[:, 1]]
    n = np.cross(a, b)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    q = p[None, :] - (n @ p)[:, None] * n
    qn = np.linalg.norm(q, axis=1)
    q = q / np.where(qn > 1e-12, qn, 1.0)[:, None]
    inside = (np.einsum("ij,ij->i", np.cross(a, q), n) >= 0) & (np.einsum("ij,ij->i", np.cross(q, b), n) >= 0) & (qn > 1e-12)
    da, db = a @ p, b @ p
    cand = np.where(inside[:, None], q, np.where((da >= db)[:, None], a, b))
    score = cand @ p
    e = int(np.argmax(score))
    qe = cand[e]
    i, j = (int(x) for x in tr.boundary_edges[e])
    # Linear-combination coefficients of qe in span(v_i, v_j) (least squares, 2 unknowns).
    vi, vj = tr.vertices[i], tr.vertices[j]
    g = np.array([[vi @ vi, vi @ vj], [vi @ vj, vj @ vj]])
    coef = np.linalg.solve(g, np.array([vi @ qe, vj @ qe]))
    coef = np.clip(coef, 0.0, None)
    s = coef.sum()
    alpha = float(coef[1] / s) if s > 0 else 0.0
    return qe, i, j, alpha


def lateral_crossings(tr: Triangulation, p: np.ndarray) -> list[tuple[float, int, int, float]]:
    """Points where the circle of constant lateral angle through p crosses boundary edges.

    Returns a list of (polar_deg, i, j, alpha) with the crossing point ~ (1-alpha) v_i + alpha v_j.
    The arc from v_i to v_j is q(t) = v_i cos t + c sin t (t in [0, T]); its y coordinate
    equals s = p_y where R cos(t - phi) = s, R = hypot(v_iy, c_y), phi = atan2(c_y, v_iy).
    Mirrors lateralCrossings() in frontend/src/dsp/interpolation.ts.
    """
    s = float(p[1])
    out = []
    for i, j in tr.boundary_edges:
        a, b = tr.vertices[i], tr.vertices[j]
        cosT = float(np.clip(a @ b, -1.0, 1.0))
        T = float(np.arccos(cosT))
        c = b - cosT * a
        cn = float(np.linalg.norm(c))
        if cn < 1e-12:
            continue
        c = c / cn
        R = float(np.hypot(a[1], c[1]))
        if R < 1e-12 or abs(s) > R:
            continue
        phi = float(np.arctan2(c[1], a[1]))
        d = float(np.arccos(np.clip(s / R, -1.0, 1.0)))
        for t in (phi - d, phi + d):
            t = (t + np.pi) % (2 * np.pi) - np.pi          # wrap to [-pi, pi)
            if -1e-12 <= t <= T + 1e-12:
                q = a * np.cos(t) + c * np.sin(t)
                # q = ka*a + kb*b exactly (q lies in span(a, b)); normalise weights to sum 1.
                kb = np.sin(t) / np.sin(T)
                ka = np.cos(t) - kb * cosT
                ka, kb = max(ka, 0.0), max(kb, 0.0)
                alpha = kb / (ka + kb) if ka + kb > 0 else 0.0
                pol = float(np.rad2deg(np.arctan2(q[2], q[0])))
                out.append((pol, int(i), int(j), float(alpha)))
    return out


def gap_fill(tr: Triangulation, p: np.ndarray):
    """Gap-fill weights for a direction inside an unmeasured region (see module docs).

    Returns (indices, weights, info) or None when the lateral circle does not bracket p.
    """
    cr = lateral_crossings(tr, p)
    if len(cr) < 2:
        return None
    beta = float(np.rad2deg(np.arctan2(p[2], p[0])))
    # Circular distance (degrees, [0, 360)) going "down" and "up" from beta to each crossing.
    down = [((beta - c[0]) % 360.0, k) for k, c in enumerate(cr)]
    up = [((c[0] - beta) % 360.0, k) for k, c in enumerate(cr)]
    dd, klo = min(down)
    du, khi = min(up)
    if klo == khi or dd + du <= 1e-9:
        return None
    t = dd / (dd + du)
    lo, hi = cr[klo], cr[khi]
    w: dict[int, float] = {}
    for (pol, i, j, al), wt in ((lo, 1.0 - t), (hi, t)):
        w[i] = w.get(i, 0.0) + wt * (1.0 - al)
        w[j] = w.get(j, 0.0) + wt * al
    idx = np.array(sorted(w), dtype=np.int64)
    wts = np.array([w[k] for k in idx])
    keep = wts > 0
    idx, wts = idx[keep], wts[keep]
    info = {"lateral_deg": float(np.rad2deg(np.arcsin(np.clip(p[1], -1, 1)))), "polar_deg": beta,
            "polar_lo_deg": lo[0], "polar_hi_deg": hi[0], "t": float(t), "gap_span_deg": float(dd + du)}
    return idx, wts / wts.sum(), info


@dataclass
class Weights:
    indices: np.ndarray            # vertex indices
    weights: np.ndarray            # normalised, sum to 1
    status: str                    # "measured" | "interpolated" | "gap_fill" | "clamped"
    rendered_direction: np.ndarray  # unit vector actually rendered
    triangle: int = -1
    max_edge_deg: float = 0.0
    nearest_index: int = -1
    nearest_angle_deg: float = 0.0
    extra: dict = field(default_factory=dict)


class DirectionWeighter:
    """Computes interpolation weights for a direction with a chosen scheme."""

    def __init__(self, vectors: np.ndarray, scheme: str = "barycentric", gap_edge_threshold_deg: float = 30.0,
                 measured_snap_deg: float = 0.1, idw_k: int = 4, idw_power: float = 2.0,
                 triangulation: Triangulation | None = None):
        if scheme not in ("nearest", "idw", "barycentric"):
            raise ValueError(f"Unknown weighting scheme '{scheme}'")
        self.vectors = np.asarray(vectors, dtype=float)
        self.vectors = self.vectors / np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.scheme = scheme
        self.snap = measured_snap_deg
        self.k = idw_k
        self.power = idw_power
        self.tri = triangulation
        if scheme == "barycentric" and self.tri is None:
            self.tri = build_triangulation(self.vectors, gap_edge_threshold_deg)

    def _nearest(self, p: np.ndarray) -> tuple[int, float]:
        dots = self.vectors @ p
        i = int(np.argmax(dots))
        return i, float(angular_distance_deg(self.vectors[i], p))

    def weights(self, direction: np.ndarray) -> Weights:
        p = np.asarray(direction, dtype=float)
        p = p / np.linalg.norm(p)
        ni, nang = self._nearest(p)
        if self.scheme == "nearest" or nang <= self.snap:
            status = "measured"
            return Weights(np.array([ni]), np.array([1.0]), status, self.vectors[ni], nearest_index=ni,
                           nearest_angle_deg=nang)
        if self.scheme == "idw":
            ang = angular_distance_deg(self.vectors, p[None, :])
            idx = np.argsort(ang, kind="stable")[: self.k]
            w = 1.0 / np.maximum(ang[idx], 1e-9) ** self.power
            return Weights(idx, w / w.sum(), "interpolated", p, nearest_index=ni, nearest_angle_deg=nang)
        t, bw = locate(self.tri, p)
        if self.tri.is_gap[t]:
            gf = gap_fill(self.tri, p)
            if gf is not None:
                idx, w, info = gf
                return Weights(idx, w, "gap_fill", p, triangle=t, max_edge_deg=float(self.tri.max_edge_deg[t]),
                               nearest_index=ni, nearest_angle_deg=nang, extra=info)
            q, i, j, alpha = clamp_to_boundary(self.tri, p)
            idx = np.array([i, j])
            w = np.array([1.0 - alpha, alpha])
            return Weights(idx, w, "clamped", q, triangle=t, max_edge_deg=float(self.tri.max_edge_deg[t]),
                           nearest_index=ni, nearest_angle_deg=nang,
                           extra={"clamp_angle_deg": float(angular_distance_deg(q, p))})
        bw = np.clip(bw, 0.0, None)
        bw = bw / bw.sum()
        return Weights(self.tri.triangles[t].copy(), bw, "interpolated", p, triangle=t,
                       max_edge_deg=float(self.tri.max_edge_deg[t]), nearest_index=ni, nearest_angle_deg=nang)


METHODS = {
    # name: (weighting scheme, representation)
    "nearest": ("nearest", "raw"),
    "idw_raw": ("idw", "raw"),
    "idw_aligned": ("idw", "aligned"),
    "barycentric_raw": ("barycentric", "raw"),
    "barycentric_aligned": ("barycentric", "aligned"),
    "barycentric_minphase": ("barycentric", "minphase"),
}


@dataclass
class InterpolatedHRIR:
    hrir: np.ndarray          # (2, L)
    toa: np.ndarray           # (2,) onset used (samples)
    w: Weights
    method: str


class HRIRInterpolator:
    """HRIR estimation for arbitrary directions from a :class:`PreparedFilters` set."""

    def __init__(self, pf: PreparedFilters, method: str = "barycentric_aligned",
                 gap_edge_threshold_deg: float = 30.0, measured_snap_deg: float = 0.1,
                 idw_k: int = 4, idw_power: float = 2.0, triangulation: Triangulation | None = None,
                 cache_size: int = 4096, direction_quantum_deg: float = 0.1):
        if method not in METHODS:
            raise ValueError(f"Unknown interpolation method '{method}'. Choose from {sorted(METHODS)}")
        self.pf = pf
        self.method = method
        scheme, self.representation = METHODS[method]
        self.weighter = DirectionWeighter(pf.vectors, scheme, gap_edge_threshold_deg, measured_snap_deg,
                                          idw_k, idw_power, triangulation)
        self.quantum = direction_quantum_deg
        self._cached = lru_cache(maxsize=cache_size)(self._compute_quantized)

    @property
    def triangulation(self) -> Triangulation | None:
        return self.weighter.tri

    def combine(self, w: Weights) -> InterpolatedHRIR:
        pf = self.pf
        idx, wt = w.indices, w.weights
        if w.status == "measured":
            i = int(idx[0])
            return InterpolatedHRIR(pf.measured[i].copy(), pf.toa[i].copy(), w, self.method)
        toa = np.einsum("k,ke->e", wt, pf.toa[idx])
        if self.representation == "raw":
            h = np.einsum("k,kel->el", wt, pf.measured[idx])
        elif self.representation == "aligned":
            shape = np.einsum("k,kel->el", wt, pf.aligned[idx])
            h = reconstruct(shape, toa, pf.margin, pf.filter_length)
        elif self.representation == "minphase":
            shape = np.einsum("k,kel->el", wt, pf.minphase[idx])
            h = reconstruct(shape, toa, 0, pf.filter_length)
        else:  # pragma: no cover
            raise ValueError(self.representation)
        return InterpolatedHRIR(h, toa, w, self.method)

    def query(self, direction: np.ndarray) -> InterpolatedHRIR:
        """Exact (unquantised) estimate for a head-frame direction vector."""
        return self.combine(self.weighter.weights(direction))

    def query_sph(self, az_deg: float, el_deg: float) -> InterpolatedHRIR:
        return self.query(sph_to_cart(az_deg, el_deg))

    def _compute_quantized(self, az_q: float, el_q: float) -> InterpolatedHRIR:
        return self.query(sph_to_cart(az_q, el_q))

    def query_quantized(self, direction: np.ndarray) -> InterpolatedHRIR:
        """Estimate on the cache grid (what the real-time renderers use)."""
        az, el, _ = cart_to_sph(np.asarray(direction, dtype=float))
        return self._cached(*quantize_direction(float(az), float(el), self.quantum))

    def cache_info(self):
        return self._cached.cache_info()
