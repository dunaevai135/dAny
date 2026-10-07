import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull, SphericalVoronoi


def generate_evenly_spread_points(
    n_points: int,
    iterations: int = 2000,
    step_size: float = 0.005,
    seed: int = 42,
    block: int = 256
):
    rng = np.random.default_rng(seed)
    points = rng.random((n_points, 3)) - 0.5
    points /= np.linalg.norm(points, axis=1, keepdims=True)

    eps = 1e-12

    for _ in range(iterations):
        forces = np.zeros_like(points)

        for i0 in range(0, n_points, block):
            i1 = min(i0 + block, n_points)
            pi = points[i0:i1]
            diff = pi[:, None, :] - points[None, :, :]
            d2 = np.einsum("bij,bij->bi", diff, diff) + eps

            for bi, gi in enumerate(range(i0, i1)):
                d2[bi, gi] = np.inf

            inv_r3 = 1.0 / (d2 * np.sqrt(d2))
            f = diff * (step_size * inv_r3)[:, :, None]
            forces[i0:i1] = f.sum(axis=1)

        points += forces
        points /= np.linalg.norm(points, axis=1, keepdims=True)

    return points


def stl_name(sides, seed, variant, cut=None):
    """Имя STL из dAny6.py; cut=None — CUT_FACTOR по умолчанию (1 − 1.76/sides)."""
    return f"d{sides}_s{seed}_{variant}" + (f"_c{cut:.3f}" if cut is not None else "") + ".stl"


def voronoi_rel_areas(points):
    """Площади сферических ячеек Вороного, нормированные на среднюю (сумма = n)."""
    p = np.asarray(points, dtype=float)
    return SphericalVoronoi(p).calculate_areas() * len(p) / (4 * np.pi)


def equalize_voronoi_areas(points, reg=1e-3, tol=1e-6, debug=False):
    """Сдвигает нормали так, чтобы площади сферических ячеек Вороного были равны.
    При равноудалённых срезах это равные вероятности граней (квазистатика)."""
    p0 = np.asarray(points, dtype=float)
    n = len(p0)

    def unit(x):
        p = p0 + x.reshape(n, 3)
        return p / np.linalg.norm(p, axis=1, keepdims=True)

    def residuals(x):
        # reg держит решение рядом с Томсоном (решений континуум)
        return np.concatenate([voronoi_rel_areas(unit(x)) - 1.0, reg * x])

    r = least_squares(residuals, np.zeros(3 * n), diff_step=1e-6, xtol=1e-12, ftol=1e-12)
    p = unit(r.x)

    a0, a1 = voronoi_rel_areas(p0), voronoi_rel_areas(p)
    if debug:
        shift = np.degrees(np.arccos(np.clip((p * p0).sum(1), -1, 1)))
        print(f"   Voronoi areas {a0.min():.3f}–{a0.max():.3f} -> {a1.min():.5f}–{a1.max():.5f}, "
              f"shift mean {shift.mean():.2f}° max {shift.max():.2f}°")
    if np.abs(a1 - 1).max() > tol:
        print(f"⚠️ Площади не выровнены до {tol}: max dev {np.abs(a1 - 1).max():.2e}")
    return p


def die_normals(n_faces, seed, equalize, relax_iterations=2000, relax_step=0.005, debug=False):
    vectors = generate_evenly_spread_points(n_faces, iterations=relax_iterations, step_size=relax_step, seed=seed)
    if equalize:
        vectors = equalize_voronoi_areas(vectors, debug=debug)
    return vectors


def die_hull_points(normals, radius, flat_factor, sphere_res=48, circle_res=96):
    """Точки границы тела «шар ∩ полупространства» (та же форма, что даёт Split в dAny6.py).
    Их выпуклая оболочка — меш для физики: грани плоские точно, сферические остатки
    на углах аппроксимируются сеткой."""
    n = np.asarray(normals, dtype=float)
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    d = radius * flat_factor
    tol = 1e-9 * radius

    def inside(p):
        return (p @ n.T <= d + tol).all(axis=1) & (np.linalg.norm(p, axis=1) <= radius + tol)

    pts = []

    # углы многогранника (тройные пересечения плоскостей)
    k = len(n)
    for i in range(k):
        for j in range(i + 1, k):
            for l in range(j + 1, k):
                A = n[[i, j, l]]
                if abs(np.linalg.det(A)) < 1e-9:
                    continue
                pts.append(np.linalg.solve(A, np.full(3, d)))

    # концы рёбер на сфере (прямая пересечения двух плоскостей ∩ сфера)
    for i in range(k):
        for j in range(i + 1, k):
            u = np.cross(n[i], n[j])
            s = np.linalg.norm(u)
            if s < 1e-9:
                continue
            u /= s
            # точка на прямой, ближайшая к центру
            G = np.array([[1, n[i] @ n[j]], [n[i] @ n[j], 1]])
            a, b = np.linalg.solve(G, [d, d])
            p0 = a * n[i] + b * n[j]
            h2 = radius**2 - p0 @ p0
            if h2 > 0:
                h = np.sqrt(h2)
                pts += [p0 + h * u, p0 - h * u]

    # окружности срезов
    rc2 = radius**2 - d**2
    if rc2 > 0:
        t = np.linspace(0, 2 * np.pi, circle_res, endpoint=False)
        for v in n:
            e1 = np.cross(v, [1, 0, 0] if abs(v[0]) < 0.9 else [0, 1, 0])
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(v, e1)
            pts += list(d * v + np.sqrt(rc2) * (np.outer(np.cos(t), e1) + np.outer(np.sin(t), e2)))

    # сферические остатки
    m = sphere_res * sphere_res
    i = np.arange(m) + 0.5
    phi = np.arccos(1 - 2 * i / m)
    theta = np.pi * (1 + 5**0.5) * i
    pts += list(radius * np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1))

    pts = np.array(pts)
    pts = pts[inside(pts)]
    hull = ConvexHull(pts)
    return pts[hull.vertices]


def die_mesh(normals, radius, flat_factor, **kw):
    """Выпуклый меш тела: (vertices, faces) с внешней ориентацией треугольников."""
    v = die_hull_points(normals, radius, flat_factor, **kw)
    hull = ConvexHull(v)
    f = hull.simplices.copy()
    c = v.mean(axis=0)
    nrm = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    flip = ((v[f[:, 0]] - c) * nrm).sum(1) < 0
    f[flip] = f[flip][:, ::-1]
    return v, f


def mesh_moments(v, f):
    """Объём, первый момент и тензор второго момента ∫x xᵀ dV относительно начала координат
    для замкнутого меша (знаковые тетраэдры с вершиной в начале координат)."""
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    vol6 = np.einsum("ij,ij->i", a, np.cross(b, c))
    vol = vol6.sum() / 6
    first = (vol6[:, None] * (a + b + c)).sum(0) / 24
    s = a + b + c
    # ∫ x xᵀ по тетраэдру (0,a,b,c) = V/20 · (Σ pᵢpᵢᵀ + s sᵀ)
    outer = np.einsum("ni,nj->nij", a, a) + np.einsum("ni,nj->nij", b, b) + np.einsum("ni,nj->nij", c, c)
    second = ((vol6 / 6)[:, None, None] * (outer + np.einsum("ni,nj->nij", s, s))).sum(0) / 20
    return vol, first, second


def shell_inertial(v, f, density, scale=None, infill=0.0):
    """Масса, центр масс и тензор инерции (около ЦМ) тела с плотной стенкой и ядром,
    полученным масштабированием меша на scale относительно начала координат (как core в dAny6.py).
    scale=None — сплошное тело."""
    vol, first, second = mesh_moments(v, f)
    m, p1, p2 = density * vol, density * first, density * second
    if scale is not None:
        k = (1 - infill) * density
        m -= k * vol * scale**3
        p1 -= k * first * scale**4
        p2 -= k * second * scale**5
    com = p1 / m
    c2 = p2 - m * np.outer(com, com)
    inertia = np.trace(c2) * np.eye(3) - c2
    return m, com, inertia


def read_stl(path):
    """Бинарный STL → (vertices, faces) без слияния вершин (для масс-свойств этого достаточно)."""
    data = np.fromfile(path, dtype=np.uint8)
    n = int(np.frombuffer(data[80:84], np.uint32)[0])
    if 84 + 50 * n != len(data):
        raise ValueError(f"{path}: ожидается бинарный STL")
    rec = np.frombuffer(data[84:], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n)
    v = rec["v"].reshape(-1, 3).astype(float)
    return v, np.arange(len(v)).reshape(-1, 3)
