"""Vendored reference Dual Contouring implementation.

Origin: BorisTheBrave / mc-dc — https://github.com/BorisTheBrave/mc-dc
        (dual_contour_3d.py, qef.py, utils_3d.py, common.py, settings.py)
License: public domain / CC0 per the upstream repo.

Self-contained — inlines V3/Quad/Mesh, QEF, adapt/frange, and constants so the
module imports without the upstream `settings.py` side-effects. Used only by
`visergui/dual_contouring.py`'s sanity test; never imported by the live mesh
path. The interface differs slightly from the upstream: `dual_contour_3d_ref`
takes explicit `bounds` and `cell_size` instead of reading them from a global
settings module.
"""

from __future__ import annotations

import math
import numpy as np


# ---- Constants (from upstream settings.py) -------------------------------- #

ADAPTIVE = True
CLIP = False
BOUNDARY = True
BIAS = True
BIAS_STRENGTH = 0.01
EPS = 1e-8


# ---- V3 / Quad / Mesh (from upstream utils_3d.py) ------------------------- #

class V3:
    def __init__(self, x, y, z):
        self.x = float(x); self.y = float(y); self.z = float(z)

    def normalize(self):
        d = math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
        return V3(self.x / d, self.y / d, self.z / d)


class Quad:
    def __init__(self, v1, v2, v3, v4):
        self.v1 = v1; self.v2 = v2; self.v3 = v3; self.v4 = v4

    def swap(self, swap=True):
        if swap:
            return Quad(self.v4, self.v3, self.v2, self.v1)
        return Quad(self.v1, self.v2, self.v3, self.v4)


class Mesh:
    def __init__(self, verts=None, faces=None):
        self.verts = verts or []
        self.faces = faces or []


# ---- adapt / frange (from upstream common.py) ----------------------------- #

def _adapt(v0, v1, cell_size):
    """Linear interp factor where the segment v0→v1 crosses zero, scaled by cell."""
    if ADAPTIVE:
        return (0 - v0) / (v1 - v0) * cell_size
    return 0.5 * cell_size


def _frange(start, stop, step):
    v = start
    while v < stop:
        yield v
        v += step


# ---- QEF (from upstream qef.py) ------------------------------------------- #

class _QEF:
    def __init__(self, A, b, fixed_values):
        self.A = A; self.b = b; self.fixed_values = fixed_values

    def evaluate(self, x):
        return float(np.linalg.norm(np.matmul(self.A, np.array(x)) - self.b))

    def eval_with_pos(self, x):
        return self.evaluate(x), list(x)

    @staticmethod
    def make_3d(positions, normals):
        A = np.array(normals)
        b = [v[0]*n[0] + v[1]*n[1] + v[2]*n[2] for v, n in zip(positions, normals)]
        fixed_values = [None] * A.shape[1]
        return _QEF(A, b, fixed_values)

    def fix_axis(self, axis, value):
        b = np.array(self.b) - self.A[:, axis] * value
        A = np.delete(self.A, axis, 1)
        fixed_values = list(self.fixed_values)
        fixed_values[axis] = value
        return _QEF(A, b, fixed_values)

    def solve(self):
        result, residual, rank, s = np.linalg.lstsq(self.A, self.b, rcond=None)
        if len(residual) == 0:
            residual = self.evaluate(result)
        else:
            residual = float(residual[0])
        position = []
        i = 0
        for value in self.fixed_values:
            if value is None:
                position.append(float(result[i])); i += 1
            else:
                position.append(float(value))
        return residual, position


def _solve_qef_3d(x, y, z, positions, normals, cell_size):
    if BIAS:
        mass_point = np.mean(positions, axis=0)
        normals = list(normals) + [
            [BIAS_STRENGTH, 0, 0], [0, BIAS_STRENGTH, 0], [0, 0, BIAS_STRENGTH]
        ]
        positions = list(positions) + [mass_point, mass_point, mass_point]

    qef = _QEF.make_3d(positions, normals)
    residual, v = qef.solve()

    if BOUNDARY:
        def inside(r):
            return (x <= r[1][0] <= x + cell_size
                    and y <= r[1][1] <= y + cell_size
                    and z <= r[1][2] <= z + cell_size)
        if not inside((residual, v)):
            rs = [qef.fix_axis(0, x + 0).solve(),
                  qef.fix_axis(0, x + cell_size).solve(),
                  qef.fix_axis(1, y + 0).solve(),
                  qef.fix_axis(1, y + cell_size).solve(),
                  qef.fix_axis(2, z + 0).solve(),
                  qef.fix_axis(2, z + cell_size).solve()]
            rs = [r for r in rs if inside(r)]
            if not rs:
                corners = [(x + dx*cell_size, y + dy*cell_size, z + dz*cell_size)
                           for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]
                rs = [qef.eval_with_pos(c) for c in corners]
                rs = [r for r in rs if inside(r)]
            if rs:
                residual, v = min(rs, key=lambda r: r[0])

    if CLIP:
        v[0] = float(np.clip(v[0], x, x + cell_size))
        v[1] = float(np.clip(v[1], y, y + cell_size))
        v[2] = float(np.clip(v[2], z, z + cell_size))

    return V3(v[0], v[1], v[2])


# ---- DC core (from upstream dual_contour_3d.py) --------------------------- #

def _find_best_vertex(f, f_normal, x, y, z, cell_size):
    if not ADAPTIVE:
        return V3(x + 0.5*cell_size, y + 0.5*cell_size, z + 0.5*cell_size)

    v = np.empty((2, 2, 2))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                v[dx, dy, dz] = f(x + dx*cell_size, y + dy*cell_size, z + dz*cell_size)

    changes = []
    for dx in (0, 1):
        for dy in (0, 1):
            if (v[dx, dy, 0] > 0) != (v[dx, dy, 1] > 0):
                changes.append((x + dx*cell_size,
                                y + dy*cell_size,
                                z + _adapt(v[dx, dy, 0], v[dx, dy, 1], cell_size)))
    for dx in (0, 1):
        for dz in (0, 1):
            if (v[dx, 0, dz] > 0) != (v[dx, 1, dz] > 0):
                changes.append((x + dx*cell_size,
                                y + _adapt(v[dx, 0, dz], v[dx, 1, dz], cell_size),
                                z + dz*cell_size))
    for dy in (0, 1):
        for dz in (0, 1):
            if (v[0, dy, dz] > 0) != (v[1, dy, dz] > 0):
                changes.append((x + _adapt(v[0, dy, dz], v[1, dy, dz], cell_size),
                                y + dy*cell_size,
                                z + dz*cell_size))

    if len(changes) <= 1:
        return None

    positions = changes
    normals = []
    for p in changes:
        n = f_normal(p[0], p[1], p[2])
        normals.append([n.x, n.y, n.z])

    return _solve_qef_3d(x, y, z, positions, normals, cell_size)


def dual_contour_3d_ref(f, f_normal, xmin, xmax, ymin, ymax, zmin, zmax,
                        cell_size=1.0):
    """Reference DC implementation. `f` is a scalar SDF (>0 outside, <0 inside),
    `f_normal` returns a V3 (gradient direction). Returns `(verts, faces)` where
    `verts` is an (N, 3) float64 array and `faces` is an (M, 4) int array of
    1-indexed quads (matching upstream's OBJ convention). Convert to 0-indexed
    + triangulate at the call site if needed.
    """
    vert_array = []
    vert_indices: dict[tuple[int, int, int], int] = {}
    xs = list(_frange(xmin, xmax, cell_size))
    ys = list(_frange(ymin, ymax, cell_size))
    zs = list(_frange(zmin, zmax, cell_size))
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            for iz, z in enumerate(zs):
                vert = _find_best_vertex(f, f_normal, x, y, z, cell_size)
                if vert is None:
                    continue
                vert_array.append(vert)
                vert_indices[(ix, iy, iz)] = len(vert_array)
    faces: list[Quad] = []
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            for iz, z in enumerate(zs):
                if x > xmin and y > ymin:
                    s1 = f(x, y, z) > 0
                    s2 = f(x, y, z + cell_size) > 0
                    if s1 != s2:
                        try:
                            faces.append(Quad(
                                vert_indices[(ix-1, iy-1, iz)],
                                vert_indices[(ix-0, iy-1, iz)],
                                vert_indices[(ix-0, iy-0, iz)],
                                vert_indices[(ix-1, iy-0, iz)],
                            ).swap(s2))
                        except KeyError:
                            pass
                if x > xmin and z > zmin:
                    s1 = f(x, y, z) > 0
                    s2 = f(x, y + cell_size, z) > 0
                    if s1 != s2:
                        try:
                            faces.append(Quad(
                                vert_indices[(ix-1, iy, iz-1)],
                                vert_indices[(ix-0, iy, iz-1)],
                                vert_indices[(ix-0, iy, iz-0)],
                                vert_indices[(ix-1, iy, iz-0)],
                            ).swap(s1))
                        except KeyError:
                            pass
                if y > ymin and z > zmin:
                    s1 = f(x, y, z) > 0
                    s2 = f(x + cell_size, y, z) > 0
                    if s1 != s2:
                        try:
                            faces.append(Quad(
                                vert_indices[(ix, iy-1, iz-1)],
                                vert_indices[(ix, iy-0, iz-1)],
                                vert_indices[(ix, iy-0, iz-0)],
                                vert_indices[(ix, iy-1, iz-0)],
                            ).swap(s2))
                        except KeyError:
                            pass

    verts_np = np.array([[v.x, v.y, v.z] for v in vert_array], dtype=np.float64)
    faces_np = np.array([[q.v1, q.v2, q.v3, q.v4] for q in faces], dtype=np.int64)
    return verts_np, faces_np


__all__ = ["dual_contour_3d_ref", "V3"]
