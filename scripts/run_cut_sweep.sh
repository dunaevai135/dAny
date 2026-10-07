#!/usr/bin/env bash
# Развёртка по CUT_FACTOR на кубиках с гравировкой: STL → симуляция → отчёт.
# Повторный запуск дописывает новые броски в ту же папку (готовые STL не пересчитываются).
#
#   DICE_FONT=/path/to/TSCu_Comic.ttf scripts/run_cut_sweep.sh
#   THROWS_PER_CUT=40000 WORKERS=32 scripts/run_cut_sweep.sh
#   RENDERS_PER_CUT=0 scripts/run_cut_sweep.sh     # без видео
set -euo pipefail
cd "$(dirname "$0")/.."

SIDES=${SIDES:-14}
SEED=${SEED:-777}
VARIANT=${VARIANT:-thomson}
CUTS=${CUTS:-0.70,0.705,0.71,0.72,0.74,0.77,0.80,0.83,0.86,0.874,0.88,0.89,0.90,0.91,0.92}
THROWS_PER_CUT=${THROWS_PER_CUT:-20000}
MASS=${MASS:-solid}
STL_DIR=${STL_DIR:-stl_cut}
OUT=${OUT:-runs/d${SIDES}_s${SEED}_cut_stl}
RENDERS_PER_CUT=${RENDERS_PER_CUT:-2}
# без дисплея нужен EGL (GPU) или osmesa (CPU): MUJOCO_GL=osmesa
export MUJOCO_GL=${MUJOCO_GL:-egl}
export DICE_FONT=${DICE_FONT:-/usr/share/fonts/TTF/TSCu_Comic.ttf}

# без шрифта dAny6.py молча возьмёт Arial — цифры и выпуклая оболочка будут другими
if [[ ! -f "$DICE_FONT" ]]; then
    echo "Нет шрифта $DICE_FONT — укажите DICE_FONT=/путь/к/TSCu_Comic.ttf" >&2
    exit 1
fi

n_cuts=$(tr ',' '\n' <<< "$CUTS" | wc -l)
throws=$((THROWS_PER_CUT * n_cuts))

uv sync
echo "== STL: $n_cuts шт. в $STL_DIR"
uv run python dAny6.py --sides "$SIDES" --seed "$SEED" --variants "$VARIANT" \
    --cut "$CUTS" --out-dir "$STL_DIR" --skip-existing
echo "== Симуляция: $throws бросков → $OUT"
uv run dice-sim run --sides "$SIDES" --seed "$SEED" --variants "$VARIANT" --stl-dir "$STL_DIR" \
    --flat "$CUTS" --mass "$MASS" --throws "$throws" --out "$OUT" ${WORKERS:+--workers "$WORKERS"}
echo "== Отчёт"
uv run dice-sim report "$OUT"

if (( RENDERS_PER_CUT > 0 )); then
    echo "== Видео: по $RENDERS_PER_CUT на CUT_FACTOR → $OUT/render"
    # конфигурации перебираются по кругу, так что на каждый CUT придётся ровно RENDERS_PER_CUT
    uv run dice-sim render --sides "$SIDES" --seed "$SEED" --variants "$VARIANT" --stl-dir "$STL_DIR" \
        --flat "$CUTS" --mass "$MASS" --count $((RENDERS_PER_CUT * n_cuts)) --out "$OUT/render" \
        || echo "⚠️ Видео не получились (проверьте MUJOCO_GL=egl|osmesa); статистика уже посчитана." >&2
fi
