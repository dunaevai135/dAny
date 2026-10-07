"""Монте-Карло бросков кубика в MuJoCo: как трение и отскок влияют на распределение граней.

    uv run dice-sim run --sides 14 --throws 200000
    uv run dice-sim report runs/d14_s777
    uv run dice-sim render --sides 14 --count 2
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .geometry import die_mesh, die_normals, read_stl, shell_inertial, stl_name, voronoi_rel_areas

RADIUS_MM = 15.0
DENSITY = 1240.0  # PLA, сплошное заполнение
TIMECONST = 0.002
REST_V = 0.005
REST_W = 0.3
REST_TIME = 0.15
FLAT_COS = np.cos(np.radians(2.0))
# Ниже этой скорости точек кубика контакт критически демпфирован (как restitution threshold
# в Bullet/PhysX): иначе пружинящий контакт на плоской грани раскачивается без конца.
BOUNCE_V = 0.15
TORSION_PER_ROLL = 5.0


@dataclass
class Shape:
    name: str
    normals: np.ndarray
    mass: float
    com: np.ndarray
    inertia: np.ndarray
    verts: np.ndarray | None = None
    faces: np.ndarray | None = None
    stl: str | None = None
    flat: float | None = None


def _parse_mass(spec):
    """'solid' или 'shell:T[:INFILL]' — стенка T мм, внутри доля заполнения INFILL (0 — пусто)."""
    if spec == "solid":
        return None, 0.0
    kind, *rest = spec.split(":")
    if kind != "shell" or not rest:
        raise ValueError(f"модель массы: solid | shell:T[:INFILL], а не {spec!r}")
    return float(rest[0]), float(rest[1]) if len(rest) > 1 else 0.0


def make_shapes(sides, seed, variants, stl_dir=None, masses=("solid",), sphere_res=64, flats=None):
    """Все сочетания вариант формы × CUT_FACTOR × модель массы.
    flats=None — один CUT_FACTOR по умолчанию, и он не попадает в имя конфигурации."""
    r = RADIUS_MM / 1000
    shapes = []
    for var in variants:
        normals = die_normals(sides, seed, equalize=var == "equal")
        for cut in flats or [None]:
            flat = cut or 1 - 1.76 / sides
            stl = None
            verts, faces = die_mesh(normals, r, flat, sphere_res=sphere_res, circle_res=2 * sphere_res)
            mv, mf = verts, faces
            if stl_dir:
                stl = str(Path(stl_dir) / stl_name(sides, seed, var, cut))
                if not os.path.exists(stl):
                    raise FileNotFoundError(f"{stl}: сначала сгенерируйте его через dAny6.py")
                mv, mf = read_stl(stl)
                mv = mv / 1000
                verts = faces = None
            for spec in masses:
                wall, infill = _parse_mass(spec)
                # ядро — масштабированная копия, стенка толщиной wall на гранях (на скруглениях чуть толще)
                scale = None if wall is None else 1 - wall / (RADIUS_MM * flat)
                m, com, inertia = shell_inertial(mv, mf, DENSITY, scale, infill)
                name = f"{var}{'-stl' if stl else ''}/{spec}" + (f"@{flat:.3f}" if cut is not None else "")
                shapes.append(Shape(name, normals, m, com, inertia, verts, faces, stl, flat))
    return shapes


# -------------------- модель --------------------

def _mesh_xml(i, shape):
    # inertia="convex": инерция меша не используется, она задана явно в <inertial>
    if shape.stl:
        return f'<mesh name="m{i}" file="{os.path.abspath(shape.stl)}" scale=".001 .001 .001" inertia="convex"/>'
    vs = " ".join(f"{x:.7g}" for x in shape.verts.ravel())
    fs = " ".join(map(str, shape.faces.ravel()))
    return f'<mesh name="m{i}" vertex="{vs}" face="{fs}" inertia="convex"/>'


def _inertial_xml(s):
    I = s.inertia
    return (f'<inertial pos="{s.com[0]:.9g} {s.com[1]:.9g} {s.com[2]:.9g}" mass="{s.mass:.9g}" '
            f'fullinertia="{I[0, 0]:.9g} {I[1, 1]:.9g} {I[2, 2]:.9g} {I[0, 1]:.9g} {I[0, 2]:.9g} {I[1, 2]:.9g}"/>')


def build_model(shapes, slots, timestep):
    """slots кубиков, распределённых по конфигурациям по кругу. Кубики друг с другом не сталкиваются."""
    assets = "\n".join(_mesh_xml(i, s) for i, s in enumerate(shapes))
    bodies = []
    for k in range(slots):
        bodies.append(f"""
        <body name="d{k}" pos="{(k % 16) * 0.1} {(k // 16) * 0.1} 0.05">
          <freejoint/>
          {_inertial_xml(shapes[k % len(shapes)])}
          <geom type="mesh" mesh="m{k % len(shapes)}" contype="2" conaffinity="1" priority="1" condim="6"
                material="die"/>
        </body>""")
    xml = f"""
    <mujoco>
      <option timestep="{timestep}" integrator="implicitfast" cone="elliptic"/>
      <visual><headlight ambient=".15 .15 .15" diffuse=".3 .3 .3" specular="0 0 0"/></visual>
      <asset>{assets}
        <texture name="grid" type="2d" builtin="checker" rgb1=".42 .45 .5" rgb2=".34 .37 .42" width="256" height="256"/>
        <material name="floor" texture="grid" texrepeat="40 40"/>
        <material name="die" rgba=".9 .55 .2 1" specular=".2" shininess=".3"/>
      </asset>
      <worldbody>
        <light pos=".5 -.5 2" dir="-.25 .25 -1" directional="true" castshadow="false" diffuse=".6 .6 .6"/>
        <geom name="floor" type="plane" size="0 0 1" contype="1" conaffinity="2" friction="0 0 0" material="floor"/>
        {''.join(bodies)}
      </worldbody>
    </mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def quat_to_zrow(q):
    """Третья строка матрицы поворота = мировая ось z в системе тела."""
    w, x, y, z = q.T
    return np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1)


def random_quat(rng, n):
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


# -------------------- калибровка отскока --------------------

def _face_down_quat(n):
    """Поворот, переводящий нормаль тела n в мировую -z."""
    z = np.array([0, 0, -1.0])
    axis = np.cross(n, z)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        return np.array([1.0, 0, 0, 0]) if n @ z > 0 else np.array([0, 1.0, 0, 0])
    ang = np.arctan2(s, n @ z)
    return np.r_[np.cos(ang / 2), np.sin(ang / 2) * axis / s]


def calibrate_restitution(shape, timestep, n=32, faces=6):
    """Таблица dampratio → эффективный коэффициент отскока при падении гранью вниз.
    MuJoCo задаёт контакт как пружину-демпфер (timeconst, dampratio), а не через e."""
    zetas = np.geomspace(0.08, 1.5, n)
    K = n * faces
    m = build_model([shape], K, timestep)
    d = mujoco.MjData(m)
    m.geom_solref[1:, 0] = TIMECONST
    m.geom_solref[1:, 1] = np.repeat(zetas, faces)
    m.geom_friction[1:, 0] = 0.5
    qpos = d.qpos.reshape(K, 7)
    qpos[:, 2] = 0.05
    qpos[:, 3:] = [_face_down_quat(shape.normals[i % faces]) for i in range(K)]
    v_in = np.zeros(K)
    v_out = np.zeros(K)
    hit = np.zeros(K, bool)
    for _ in range(int(0.3 / timestep)):
        mujoco.mj_step(m, d)
        vz = d.qvel.reshape(K, 6)[:, 2]
        v_in = np.minimum(v_in, vz)
        hit |= vz > 0
        v_out = np.where(hit, np.maximum(v_out, vz), v_out)
    e = (v_out / -v_in).reshape(n, faces).mean(1)
    # при малом демпфировании отскок шумит из-за переключения точек контакта на плоской грани
    e = np.maximum.accumulate(e[::-1])[::-1]
    keep = e < 0.95
    return zetas[keep], e[keep]


def zeta_for(e, table):
    zetas, es = table
    return np.interp(e, es[::-1], zetas[::-1])


# -------------------- воркер --------------------

def _parse_range(s):
    """'a:b' — равномерно на отрезке, 'a,b,c' — случайный выбор из уровней (сетка)."""
    if ":" in s:
        lo, hi = map(float, s.split(":"))
        return lambda rng, n: rng.uniform(lo, hi, n)
    levels = np.array([float(x) for x in s.split(",")])
    return lambda rng, n: rng.choice(levels, n)


_W = {}


def _warn_e_range(spec, tables):
    vals = [float(x) for x in spec.replace(":", ",").split(",")]
    lo, hi = max(t[1].min() for t in tables), min(t[1].max() for t in tables)
    if min(vals) < lo or max(vals) > hi:
        print(f"⚠️ e вне достижимого диапазона [{lo:.2f}, {hi:.2f}] при этом шаге — будет обрезано. "
              f"Больший e требует меньшего --timestep.")


def _worker_init(shapes, cfg, tables):
    m = build_model(shapes, cfg["slots"], cfg["timestep"])
    _W.update(m=m, d=mujoco.MjData(m), shapes=shapes, cfg=cfg, tables=tables)


RESULT_DTYPE = [
    ("shape", "i1"), ("mu", "f4"), ("e", "f4"), ("roll", "f4"), ("h", "f4"), ("v", "f4"), ("w", "f4"),
    ("bottom", "i1"), ("top", "i1"), ("align", "f4"), ("t", "f4"), ("bounces", "i2"),
]


def run_chunk(n_throws, seed):
    m, d, shapes, cfg, tables = _W["m"], _W["d"], _W["shapes"], _W["cfg"], _W["tables"]
    rng = np.random.default_rng(seed)
    K = cfg["slots"]
    sample_mu = _parse_range(cfg["mu"])
    sample_e = _parse_range(cfg["e"])
    sample_roll = _parse_range(cfg["roll"])
    normals = np.stack([shapes[k % len(shapes)].normals for k in range(K)])  # (K, F, 3)
    slot_shape = np.arange(K) % len(shapes)
    dt = m.opt.timestep
    sub = cfg["check_every"]
    rest_need = int(np.ceil(REST_TIME / (dt * sub)))
    t_max = cfg["t_max"]

    mujoco.mj_resetData(m, d)
    qpos = d.qpos.reshape(K, 7)
    qvel = d.qvel.reshape(K, 6)

    # равная квота на конфигурацию: иначе медленно успокаивающиеся формы недобирают бросков
    C = len(shapes)
    quota = -(-n_throws // C)
    n_throws = quota * C
    started_c = np.zeros(C, int)
    out = np.zeros(n_throws, RESULT_DTYPE)
    started = 0
    done = 0
    cur = np.full(K, -1)          # индекс броска в out или -1 (слот свободен)
    t0 = np.zeros(K)
    still = np.zeros(K, int)
    bounces = np.zeros(K, int)
    was_up = np.zeros(K, bool)
    zeta = np.ones(K)
    rad = RADIUS_MM / 1000
    t = 0.0

    def launch(ks):
        nonlocal started
        keep = []
        for k in ks:
            c = slot_shape[k]
            if started_c[c] < quota:
                started_c[c] += 1
                keep.append(k)
        ks = np.array(keep, int)
        if len(ks) == 0:
            return
        n = len(ks)
        idx = np.arange(started, started + n)
        started += n
        mu, e, roll = sample_mu(rng, n), sample_e(rng, n), sample_roll(rng, n)
        h = rng.uniform(*cfg["height"], n)
        v = rng.uniform(*cfg["speed"], n)
        w = rng.uniform(*cfg["spin"], n)
        phi = rng.uniform(0, 2 * np.pi, n)
        vz = -rng.uniform(0, 1, n) * v * 0.5
        wd = rng.normal(size=(n, 3))
        wd /= np.linalg.norm(wd, axis=1, keepdims=True)

        qpos[ks, 2] = h
        qpos[ks, 3:] = random_quat(rng, n)
        qvel[ks, 0] = v * np.cos(phi)
        qvel[ks, 1] = v * np.sin(phi)
        qvel[ks, 2] = vz
        qvel[ks, 3:] = wd * w[:, None]
        g = ks + 1  # geom 0 — пол
        m.geom_friction[g] = np.stack([mu, roll * TORSION_PER_ROLL, roll], 1)
        m.geom_solref[g, 0] = TIMECONST
        for c in np.unique(slot_shape[ks]):
            sel = slot_shape[ks] == c
            zeta[ks[sel]] = zeta_for(e[sel], tables[c])

        o = out[idx]
        o["shape"], o["mu"], o["e"], o["roll"] = slot_shape[ks], mu, e, roll
        o["h"], o["v"], o["w"] = h, v, w
        out[idx] = o
        cur[ks] = idx
        t0[ks] = t
        still[ks] = 0
        bounces[ks] = 0
        was_up[ks] = False

    launch(np.arange(K))
    mujoco.mj_forward(m, d)

    while done < n_throws:
        mujoco.mj_step(m, d, nstep=sub)
        t += dt * sub
        active = cur >= 0

        up = qvel[:, 2] > 0.05
        bounces += up & ~was_up & active
        was_up = up

        lin = np.linalg.norm(qvel[:, :3], axis=1)
        ang = np.linalg.norm(qvel[:, 3:], axis=1)
        m.geom_solref[1:, 1] = np.where(lin + ang * rad < BOUNCE_V, 1.0, zeta)
        down = -quat_to_zrow(qpos[:, 3:])
        dots = np.einsum("kfi,ki->kf", normals, down)
        bottom = dots.argmax(1)
        align = dots.max(1)
        calm = (lin < REST_V) & (ang < REST_W)
        still = np.where(calm, still + 1, 0)

        # покой не на грани — кубик замер на сферическом остатке угла (там безразличное равновесие)
        finished = active & ((still >= rest_need) | (t - t0 > t_max))
        ks = np.where(finished)[0]
        if len(ks):
            idx = cur[ks]
            ok = (still[ks] >= rest_need) & (align[ks] > FLAT_COS)
            o = out[idx]
            o["bottom"] = np.where(ok, bottom[ks], -1)
            o["top"] = np.where(ok, dots[ks].argmin(1), -1)
            o["align"] = align[ks]
            o["t"] = t - t0[ks]
            o["bounces"] = bounces[ks]
            out[idx] = o
            done += len(ks)
            cur[ks] = -1
            # без новых бросков слот так и лежит на столе, это почти бесплатно
            launch(ks)
    return out


# -------------------- CLI --------------------

def _pair(s):
    a, b = map(float, s.split(":"))
    return a, b


def cmd_run(a):
    shapes = _shapes_from_args(a)
    out_dir = Path(a.out or f"runs/d{a.sides}_s{a.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Конфигурации и калибровка отскока:")
    tables = []
    for s in shapes:
        tables.append(calibrate_restitution(s, a.timestep))
        print(f"   {s.name:24s} m={s.mass * 1e3:5.2f} г  I/(m r²)={np.trace(s.inertia) / 3 / s.mass / (RADIUS_MM / 1000) ** 2:.3f}"
              f"  |ЦМ|={np.linalg.norm(s.com) * 1e3:.3f} мм  e∈[{tables[-1][1].min():.2f}, {tables[-1][1].max():.2f}]")
    _warn_e_range(a.e, tables)

    cfg = dict(
        # кратно числу конфигураций, иначе первые получают больше бросков
        slots=-(-a.slots // len(shapes)) * len(shapes), timestep=a.timestep, check_every=a.check_every,
        mu=a.mu, e=a.e, roll=a.roll, height=a.height, speed=a.speed, spin=a.spin, t_max=a.t_max,
    )
    meta = dict(
        sides=a.sides, seed=a.seed, shapes=[s.name for s in shapes], stl=[s.stl for s in shapes],
        flats=[s.flat for s in shapes],
        mass=[s.mass for s in shapes], com=[s.com.tolist() for s in shapes],
        inertia=[s.inertia.tolist() for s in shapes], radius=RADIUS_MM / 1000,
        voronoi=[voronoi_rel_areas(s.normals).tolist() for s in shapes],
        normals=[s.normals.tolist() for s in shapes], cfg=cfg,
        restitution_tables=[[x.tolist() for x in t] for t in tables],
    )

    workers = a.workers or os.cpu_count()
    chunks = [a.chunk] * (a.throws // a.chunk) + ([a.throws % a.chunk] if a.throws % a.chunk else [])
    base_seed = a.run_seed if a.run_seed is not None else int(time.time())
    seeds = np.random.SeedSequence(base_seed).generate_state(len(chunks))

    print(f"Симуляция: {a.throws} бросков, {len(shapes)} конфигураций, {workers} процессов × {cfg['slots']} кубиков...")
    t0 = time.time()
    res = []
    with mp.get_context("spawn").Pool(workers, _worker_init, (shapes, cfg, tables)) as pool:
        for r in pool.imap(_star_run_chunk, zip(chunks, seeds.tolist())):
            res.append(r)
            n = sum(map(len, res))
            el = time.time() - t0
            print(f"\r   {n}/{a.throws}  {n / el:.0f} бросков/с  ETA {(a.throws - n) / max(n / el, 1e-9):.0f} с",
                  end="", flush=True)
    print()
    res = np.concatenate(res)
    fn = out_dir / f"throws_{base_seed}.npz"
    np.savez_compressed(fn, throws=res, meta=json.dumps(meta))
    bad = (res["bottom"] < 0).mean()
    print(f"✅ {fn}  ({time.time() - t0:.1f} с, не легли на грань: {bad:.2%}, "
          f"среднее время до покоя {res['t'].mean():.2f} с)")


def _shapes_from_args(a):
    flats = [float(x) for x in a.flat.split(",")] if a.flat else None
    return make_shapes(a.sides, a.seed, a.variants.split(","), a.stl_dir, a.mass.split(","), a.sphere_res, flats)


def _star_run_chunk(args):
    return run_chunk(*args)


def cmd_render(a):
    import imageio.v3 as iio

    shapes = _shapes_from_args(a)
    rng = np.random.default_rng(a.run_seed)
    out_dir = Path(a.out or f"runs/d{a.sides}_s{a.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    mu_s, e_s = _parse_range(a.mu), _parse_range(a.e)
    for i in range(a.count):
        shape = shapes[i % len(shapes)]
        table = calibrate_restitution(shape, a.timestep)
        m = build_model([shape], 1, a.timestep)
        m.vis.global_.offwidth, m.vis.global_.offheight = 640, 480
        d = mujoco.MjData(m)
        mu, e, roll = mu_s(rng, 1)[0], e_s(rng, 1)[0], _parse_range(a.roll)(rng, 1)[0]
        m.geom_friction[1] = [mu, roll * TORSION_PER_ROLL, roll]
        zeta = zeta_for(e, table)
        m.geom_solref[1] = [TIMECONST, zeta]
        d.qpos[2] = rng.uniform(*a.height)
        d.qpos[3:] = random_quat(rng, 1)[0]
        v, phi = rng.uniform(*a.speed), rng.uniform(0, 2 * np.pi)
        d.qvel[:3] = [v * np.cos(phi), v * np.sin(phi), -0.3 * v]
        wd = rng.normal(size=3)
        d.qvel[3:] = wd / np.linalg.norm(wd) * rng.uniform(*a.spin)

        frames = []
        cam = mujoco.MjvCamera()
        cam.distance, cam.elevation, cam.azimuth = 0.16, -35, 45
        fps = 60
        with mujoco.Renderer(m, 480, 640) as r:
            for k in range(int(a.duration / a.timestep)):
                mujoco.mj_step(m, d)
                slow = np.linalg.norm(d.qvel[:3]) + np.linalg.norm(d.qvel[3:]) * RADIUS_MM / 1000 < BOUNCE_V
                m.geom_solref[1, 1] = 1.0 if slow else zeta
                if k % max(1, round(a.slowmo / (fps * a.timestep))) == 0:
                    cam.lookat[:] = [d.qpos[0], d.qpos[1], max(0.01, d.qpos[2] * 0.5)]
                    r.update_scene(d, cam)
                    frames.append(r.render())
        down = -quat_to_zrow(d.qpos[3:][None])[0]
        face = int((shape.normals @ down).argmax())
        fn = out_dir / f"throw_{i}_{shape.name.replace('/', '_')}_mu{mu:.2f}_e{e:.2f}_face{face + 1}.mp4"
        iio.imwrite(fn, np.stack(frames), fps=fps)
        print(f"✅ {fn}")


def cmd_calibrate(a):
    for s in _shapes_from_args(a):
        print(s.name)
        for z, e in zip(*calibrate_restitution(s, a.timestep)):
            print(f"   dampratio {z:.3f}  →  e = {e:.3f}")


def main():
    ap = argparse.ArgumentParser(prog="dice-sim")
    sp = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--sides", type=int, default=14)
        p.add_argument("--seed", type=int, default=777, help="seed релаксации точек (как в dAny6.py)")
        p.add_argument("--variants", default="thomson,equal")
        p.add_argument("--mass", default="solid,shell:1.2",
                       help="распределение массы через запятую: solid | shell:T[:INFILL] (стенка T мм, заполнение 0..1)")
        p.add_argument("--flat", help="CUT_FACTOR через запятую (по умолчанию 1 − 1.76/sides); "
                       "с --stl-dir нужны файлы из dAny6.py --cut")
        p.add_argument("--sphere-res", type=int, default=64,
                       help="тесселяция сферических остатков (≈ sphere_res² точек на всю сферу)")
        p.add_argument("--stl-dir", help="брать d{N}_s{seed}_{variant}.stl с гравировкой вместо гладкой формы")
        p.add_argument("--timestep", type=float, default=0.00025)
        p.add_argument("--out")
        p.add_argument("--run-seed", type=int)
        p.add_argument("--mu", default="0.05:1.0", help="трение: 'a:b' равномерно или 'a,b,c' уровни")
        p.add_argument("--e", default="0.1:0.7", help="коэффициент отскока: 'a:b' или 'a,b,c'")
        p.add_argument("--roll", default="0.0002",
                       help="трение качения, м (верчения = ×5); без него кубик катается по круговой кромке грани без потерь")
        p.add_argument("--height", type=_pair, default=(0.05, 0.25), help="высота центра, м")
        p.add_argument("--speed", type=_pair, default=(0.2, 1.5), help="скорость броска, м/с")
        p.add_argument("--spin", type=_pair, default=(5.0, 50.0), help="угловая скорость, рад/с")

    p = sp.add_parser("run", help="массовая симуляция")
    common(p)
    p.add_argument("--throws", type=int, default=100_000)
    p.add_argument("--slots", type=int, default=16, help="кубиков в одном мире MuJoCo")
    p.add_argument("--chunk", type=int, default=2000)
    p.add_argument("--workers", type=int)
    p.add_argument("--check-every", type=int, default=20, help="шагов между проверками покоя")
    p.add_argument("--t-max", type=float, default=15.0)
    p.set_defaults(fn=cmd_run)

    p = sp.add_parser("render", help="видео нескольких бросков для контроля")
    common(p)
    p.add_argument("--count", type=int, default=2)
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument("--slowmo", type=float, default=0.5, help="доля реальной скорости в видео")
    p.set_defaults(fn=cmd_render)

    p = sp.add_parser("calibrate", help="таблица dampratio → коэффициент отскока")
    common(p)
    p.set_defaults(fn=cmd_calibrate)

    p = sp.add_parser("report", help="статистика и графики")
    p.add_argument("dirs", nargs="+", help="папки с throws_*.npz; несколько — объединяются")
    p.add_argument("--out", help="куда писать отчёт (по умолчанию первая папка)")
    p.add_argument("--variants", help="оставить только эти варианты, напр. thomson,thomson-stl")
    p.add_argument("--bins", type=int, default=4)
    p.add_argument("--readout", choices=["bottom", "top"], default="bottom")
    p.set_defaults(fn=lambda a: __import__("dany6_sim.report", fromlist=["report"]).report(a))

    a = ap.parse_args()
    a.fn(a)
