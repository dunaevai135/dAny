# v5: ускоренная версия 4_test.py (~25–30% быстрее, результат идентичен — тот же объём).
#
# Было: fuse(все цифры) → outer.cut → union(core).
# Стало: для каждой цифры E_i = (text_i ∩ outer) − core, затем один outer.cut(*E_i).
#   Это то же самое, т.к. core ⊂ outer, но без самого дорогого шага — union с ядром.
#   E_i независимы и считаются параллельно в процессах (pool через fork до первой булевой операции).
#
# Почти всё время — пересечение боковых граней букв с гранями кубика внутри OCC,
# поэтому параллелизм упирается в число быстрых ядер. Пробовал и отбросил:
#   - одна булева операция через BOPAlgo_CellsBuilder вместо трёх — без выигрыша;
#   - полигонизация контуров букв — сильно медленнее и невалидный результат;
#   - склейка сегментов контура в один B-сплайн — медленнее;
#   - NurbsConvert резаков, SetUseOBB, атрибуты сечения — без выигрыша;
#   - Glue в финальном cut — объём 0;
#   - потоки вместо процессов — OCP не отпускает GIL.

import cadquery as cq
import numpy as np
import os
import math
import multiprocessing as mp
from io import BytesIO
import tempfile
import contextlib
import argparse

from dany6_sim.geometry import generate_evenly_spread_points, equalize_voronoi_areas, stl_name

# --- ИМПОРТЫ OCP (шрифты) ---
from OCP.TCollection import TCollection_AsciiString
from OCP.Font import Font_FontMgr, Font_SystemFont, Font_FA_Regular

# --- ИМПОРТЫ OCP (Split для базовой формы) ---
from OCP.BOPAlgo import BOPAlgo_Splitter
from OCP.TopTools import TopTools_ListOfShape
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
from OCP.gp import gp_Pln, gp_Pnt, gp_Dir
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID, TopAbs_IN
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp


# -------------------- FONT --------------------

def register_custom_font(path_to_ttf, internal_name="DiceFont"):
    if not os.path.exists(path_to_ttf):
        print(f"⚠️ Файл не найден: {path_to_ttf}. Использую Arial.")
        return "Arial"

    try:
        mgr = Font_FontMgr.GetInstance_s()
        ocp_name = TCollection_AsciiString(internal_name)
        ocp_path = TCollection_AsciiString(path_to_ttf)

        my_font = Font_SystemFont(ocp_name)
        my_font.SetFontPath(Font_FA_Regular, ocp_path, 0)

        mgr.RegisterFont(my_font, True)
        print(f"✅ Шрифт зарегистрирован: '{internal_name}' -> {path_to_ttf}")
        return internal_name

    except Exception as e:
        print(f"❌ Не удалось зарегистрировать шрифт: {e}")
        return "Arial"


# -------------------- BASE: Split + "оставить внутреннее" --------------------

def _volume_of_solid(topods_solid) -> float:
    props = GProp_GProps()
    if hasattr(BRepGProp, "VolumeProperties_s"):
        BRepGProp.VolumeProperties_s(topods_solid, props)
    else:
        BRepGProp.VolumeProperties(topods_solid, props)
    return float(props.Mass())

def _topods_to_cq_wp(shape) -> cq.Workplane:
    from cadquery.occ_impl.shapes import Shape as CQShape
    return cq.Workplane(obj=CQShape.cast(shape))

def create_die_base_split_keep_origin(vectors, radius, flat_factor, plane_half=None, fuzzy=1e-6, classify_tol=1e-7, debug=False):
    vectors = np.asarray(vectors, dtype=float)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    cut_dist = float(radius) * float(flat_factor)
    if plane_half is None:
        plane_half = 10.0 * float(radius)

    sphere = BRepPrimAPI_MakeSphere(gp_Pnt(0.0, 0.0, 0.0), float(radius)).Shape()

    args = TopTools_ListOfShape()
    args.Append(sphere)

    tools = TopTools_ListOfShape()
    for v in vectors:
        o = v * cut_dist
        pln = gp_Pln(
            gp_Pnt(float(o[0]), float(o[1]), float(o[2])),
            gp_Dir(float(v[0]), float(v[1]), float(v[2]))
        )
        face = BRepBuilderAPI_MakeFace(
            pln,
            -float(plane_half), float(plane_half),
            -float(plane_half), float(plane_half)
        ).Face()
        tools.Append(face)

    splitter = BOPAlgo_Splitter()
    splitter.SetArguments(args)
    splitter.SetTools(tools)

    try:
        splitter.SetFuzzyValue(float(fuzzy))
    except Exception:
        pass

    splitter.Perform()
    res = splitter.Shape()

    kp = gp_Pnt(0.0, 0.0, 0.0)
    exp = TopExp_Explorer(res, TopAbs_SOLID)

    total = 0
    candidates = []
    while exp.More():
        total += 1
        solid = exp.Current()
        cls = BRepClass3d_SolidClassifier(solid, kp, float(classify_tol))
        if cls.State() == TopAbs_IN:
            candidates.append((_volume_of_solid(solid), solid))
        exp.Next()

    if debug:
        print(f"   Split solids={total}, keep(origin) candidates={len(candidates)}")

    if not candidates:
        raise RuntimeError("Split не нашёл solid, содержащий origin.")

    candidates.sort(key=lambda t: t[0], reverse=True)
    best = candidates[0][1]
    return _topods_to_cq_wp(best)


# -------------------- MAIN --------------------

def make_text_cutter(i, v, rng_angle, plane_r, text_cut_distance, font_size, font_name, random_rotation):
    number = i + 1
    text_vec = -v

    text_str = str(number) + ("." if number in [6, 9] else "")

    up_ref = np.array([0, 0, 1], dtype=float) if abs(text_vec[2]) < 0.9 else np.array([0, 1, 0], dtype=float)
    x_dir = np.cross(up_ref, text_vec)
    if np.linalg.norm(x_dir) < 1e-9:
        up_ref = np.array([1, 0, 0], dtype=float)
        x_dir = np.cross(up_ref, text_vec)
    x_dir = x_dir / np.linalg.norm(x_dir)

    origin = text_vec * plane_r
    plane = cq.Plane(origin=tuple(origin), normal=tuple(text_vec), xDir=tuple(x_dir))

    cutter = (
        cq.Workplane(plane)
        .center(-0.6, -0.8)
        .text(
            text_str,
            fontsize=font_size,
            distance=text_cut_distance,
            font=font_name,
            kind='regular',
            halign='center',
            valign='center',
            combine=True
        )
        .val()
    )

    if random_rotation:
        cutter = cutter.rotate(tuple(origin), tuple(origin + text_vec), rng_angle)
    return cutter


# Стандартный pickle (BinTools) не читает масштабированное ядро — гоняем текстовый BREP,
# а outer/core отдаём воркерам файлами.
def _to_brep(shape) -> bytes:
    buf = BytesIO()
    shape.exportBrep(buf)
    return buf.getvalue()

def _from_brep(data: bytes):
    return cq.Shape.importBrep(BytesIO(data))


_LOADED = {}

def _load(path):
    if path not in _LOADED:
        _LOADED[path] = cq.Shape.importBrep(path)
    return _LOADED[path]

def _engrave_volume(i, job, outer_path, core_path):
    cutter = make_text_cutter(
        i, job["vectors"][i], job["angles"][i], job["plane_r"], job["text_cut_distance"],
        job["font_size"], job["font_name"], job["random_rotation"]
    )
    # Обрезка по outer даёт совпадающие грани — финальный cut с ними быстрее.
    return _to_brep(cutter.intersect(_load(outer_path)).cut(_load(core_path)))


def create_polyhedral_die(
    num_faces=19,
    radius=15,
    flat_factor=0.85,
    text_depth=1.0,
    font_size=6,
    font_name="Arial",
    random_rotation=True,
    seed=42,

    # релаксация
    relax_iterations=2000,
    relax_step=0.005,
    relax_block=256,
    equalize=False,

    # split базы
    split_plane_half=None,
    split_fuzzy=1e-6,
    split_classify_tol=1e-7,
    split_debug=False,

    text_up=0.8,
    text_down=0.6,
    text_eps=0.15,

    workers=None,
):
    workers = workers or min(num_faces, os.cpu_count() or 1)
    with contextlib.ExitStack() as stack:
        pool = None
        if workers > 1:
            # fork только до первой булевой операции: иначе дети наследуют пул потоков OCC и виснут.
            pool = stack.enter_context(mp.get_context("fork").Pool(workers))
        tmp = stack.enter_context(tempfile.TemporaryDirectory())

        print(f"1. Calculating vectors for {num_faces} faces (relaxation)...")
        vectors = generate_evenly_spread_points(
            num_faces,
            iterations=relax_iterations,
            step_size=relax_step,
            seed=seed,
            block=relax_block
        )
        if equalize:
            vectors = equalize_voronoi_areas(vectors, debug=True)
        rng = np.random.default_rng(seed)
        angles = [float(rng.uniform(0, 360)) for _ in range(num_faces)]

        print("2. Generating Outer Shell (Split + keep origin)...")
        outer_die = create_die_base_split_keep_origin(
            vectors,
            radius,
            flat_factor,
            plane_half=split_plane_half,
            fuzzy=split_fuzzy,
            classify_tol=split_classify_tol,
            debug=split_debug
        ).val()

        print("3. Generating Inner Core...")
        scale_factor = (radius - text_depth) / radius
        inner_core = outer_die.scale(scale_factor)

        cut_dist = radius * flat_factor
        plane_r = radius + float(text_up)
        plane_to_face = max(plane_r - cut_dist, 0.0)
        text_cut_distance = -(plane_to_face + float(text_depth) + float(text_down) + float(text_eps))

        job = dict(
            vectors=vectors, angles=angles, plane_r=plane_r, text_cut_distance=text_cut_distance,
            font_size=font_size, font_name=font_name, random_rotation=random_rotation,
        )
        outer_path = os.path.join(tmp, "outer.brep")
        core_path = os.path.join(tmp, "core.brep")
        outer_die.exportBrep(outer_path)
        inner_core.exportBrep(core_path)

        # outer − ∪((text_i ∩ outer) − core) == (outer − ∪text_i) ∪ core, т.к. core ⊂ outer;
        # дорогой union с ядром не нужен, а работа по каждой цифре независима и параллелится.
        print(f"4. Engrave volumes per digit (font: '{font_name}', workers={workers})...")
        tasks = [(i, job, outer_path, core_path) for i in range(num_faces)]
        if pool:
            engraves = pool.starmap(_engrave_volume, tasks, chunksize=1)
        else:
            engraves = [_engrave_volume(*t) for t in tasks]

    print("5. Final cut...")
    return cq.Workplane(obj=outer_die.cut(*map(_from_brep, engraves)))


# -------------------- SETTINGS --------------------

D_SIDES = 13
TEXT_DEPTH = 0.4
FONT_SIZE = 15
RADIUS = 15.0
SEED = 777

# Укажите путь к вашему шрифту или оставьте пустым, чтобы использовался Arial
FONT_PATH = os.environ.get("DICE_FONT", "./TSCu_Comic.ttf")
# Variable-шрифт не годится: перекрывающиеся контуры глифов дают невалидные solid'ы.
# Статика: uvx --from 'fonttools[pathops]' fonttools varLib.instancer --remove-overlaps -o Cinzel-Regular-static.ttf Cinzel-VariableFont_wght.ttf wght=400
# FONT_PATH = "/home/user/Downloads/Cinzel/Cinzel-Regular-static.ttf"

if FONT_PATH != "Arial":
    REGISTERED_FONT_NAME = register_custom_font(FONT_PATH, internal_name="MyUniqueDiceFont")
else:
    REGISTERED_FONT_NAME = "Arial"


# -------------------- RUN --------------------

def render_variant(sides, seed, variant, cut=None, out_dir="."):
    try:
        result = create_polyhedral_die(
            num_faces=sides,
            radius=RADIUS,
            flat_factor=cut or 1 - 1.76 / sides,
            text_depth=TEXT_DEPTH,
            font_size=FONT_SIZE,
            font_name=REGISTERED_FONT_NAME,
            seed=seed,
            relax_iterations=2000,
            relax_step=0.005,
            equalize=variant == "equal",
            split_plane_half=12 * RADIUS,
            split_fuzzy=2e-6,
            split_classify_tol=5e-7,
            split_debug=True,
            text_up=1.0,
            text_down=1.1,
            text_eps=0.55,
        )

        filename = os.path.join(out_dir, stl_name(sides, seed, variant, cut))
        cq.exporters.export(result, filename)
        print(f"✅ Exported: {filename}")

    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sides", type=int, default=D_SIDES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--skip-existing", action="store_true", help="не пересчитывать уже готовые STL")
    ap.add_argument("--cut", help="CUT_FACTOR через запятую; по умолчанию 1 − 1.76/sides")
    ap.add_argument("--variants", default="thomson,equal",
                    help="thomson — точки Томсона, equal — с выравниванием площадей Вороного")
    args = ap.parse_args()

    # Каждый вариант в свежем процессе: fork-пул после булевых операций OCC в родителе виснет.
    os.makedirs(args.out_dir, exist_ok=True)
    cuts = [float(c) for c in args.cut.split(",")] if args.cut else [None]
    for variant in args.variants.split(","):
        for cut in cuts:
            if args.skip_existing and os.path.exists(os.path.join(args.out_dir, stl_name(args.sides, args.seed, variant, cut))):
                continue
            p = mp.get_context("spawn").Process(target=render_variant, args=(args.sides, args.seed, variant, cut, args.out_dir))
            p.start()
            p.join()
