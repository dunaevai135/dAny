"""Статистика бросков: перекос граней по конфигурациям и его зависимость от трения/отскока."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2 as chi2_dist
from scipy.stats import chi2_contingency

VARIANT_COLOR = {"thomson": "#2a78d6", "equal": "#eb6834", "thomson-stl": "#1baf7a", "equal-stl": "#e87ba4"}
MASS_STYLE = ["-", "--", ":", "-."]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
N_BOOT = 300

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "font.size": 10,
})


def load(dirs):
    files = [f for d in dirs for f in sorted(Path(d).glob("throws_*.npz"))]
    if not files:
        raise SystemExit(f"нет throws_*.npz в {', '.join(map(str, dirs))}")
    metas, parts = [], []
    for f in files:
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        t = z["throws"].copy()
        metas.append(meta)
        parts.append((meta, t))
    # прогоны могут иметь разный набор конфигураций — сводим по имени
    names = []
    for meta in metas:
        names += [n for n in meta["shapes"] if n not in names]
    base = {n: (m, i) for m in metas for i, n in enumerate(m["shapes"])}
    out = []
    for meta, t in parts:
        remap = np.array([names.index(n) for n in meta["shapes"]])
        t["shape"] = remap[t["shape"]]
        out.append(t)
    configs = []
    for n in names:
        m, i = base[n]
        normals = np.array(m["normals"][i])
        flat = (m.get("flats") or [None] * len(m["shapes"]))[i] or 1 - 1.76 / m["sides"]
        configs.append(dict(name=n, variant=n.split("/")[0], mass=n.split("/", 1)[1].split("@")[0],
                            flat=flat, normals=normals, voronoi=np.array(m["voronoi"][i])))
    if len({(m["sides"], m["seed"]) for m in metas}) > 1:
        raise SystemExit("в папках прогоны разных кубиков (sides/seed)")
    return np.concatenate(out), configs, metas[-1]


def select(T, configs, variants):
    """Оставить только конфигурации с данными вариантами формы, перенумеровав shape."""
    keep = [i for i, c in enumerate(configs) if c["variant"] in variants]
    if not keep:
        raise SystemExit(f"нет конфигураций с вариантами {variants}")
    remap = np.full(len(configs), -1)
    remap[keep] = np.arange(len(keep))
    T = T[np.isin(T["shape"], keep)].copy()
    T["shape"] = remap[T["shape"]]
    return T, [configs[i] for i in keep]


def top_map(normals):
    """Какая грань сверху, когда кубик лежит на грани j."""
    return (normals @ normals.T).argmin(1)


def quasi_probs(cfg, readout):
    a = cfg["voronoi"] / cfg["voronoi"].sum()
    if readout == "bottom":
        return a
    return np.bincount(top_map(cfg["normals"]), weights=a, minlength=len(a))


def rms_bias(counts):
    """RMS относительного отклонения p_i·N от 1 с вычтенным шумом выборки:
    E[χ²] = dof + n·Σ(p_i·N−1)²/N."""
    counts = np.atleast_2d(counts)
    n = counts.sum(-1)
    N = counts.shape[-1]
    E = n[..., None] / N
    chi2 = ((counts - E) ** 2 / np.maximum(E, 1e-12)).sum(-1)
    return np.sqrt(np.maximum(0, (chi2 - (N - 1)) / np.maximum(n, 1)))


def beta(counts, x):
    """Доля квазистатического перекоса x = a_i−1, видимая в динамике (МНК через ноль)."""
    counts = np.atleast_2d(counts)
    N = counts.shape[-1]
    y = counts / np.maximum(counts.sum(-1, keepdims=True), 1) * N - 1
    return (y * x).sum(-1) / (x @ x)


def boot(counts, fn, rng):
    """95% ДИ метрики по мультиномиальной перевыборке. Перевыборка из p̂ добавляет шум ещё раз
    и сдвигает RMS вверх, поэтому перцентили сдвигаются так, чтобы медиана совпала с оценкой."""
    n = counts.sum()
    if n == 0:
        return np.nan, np.nan, np.nan
    sims = rng.multinomial(n, counts / n, size=N_BOOT)
    est = float(fn(counts[None])[0])
    lo, med, hi = np.percentile(fn(sims), [2.5, 50, 97.5])
    return est, lo - med + est, hi - med + est


def boot_rms(counts, rng):
    est, lo, hi = boot(counts, rms_bias, rng)
    return est, max(lo, 0.0), hi


def _bins(x, k):
    u = np.unique(x)
    if len(u) <= k:
        return u, np.searchsorted(u, x), [f"{v:g}" for v in u]
    edges = np.quantile(x, np.linspace(0, 1, k + 1))
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, k - 1)
    centers = np.array([x[idx == i].mean() for i in range(k)])
    return centers, idx, [f"{edges[i]:.2g}–{edges[i + 1]:.2g}" for i in range(k)]


def rms_diff(c1, c2):
    """RMS разницы p_i·N двух конфигураций с вычтенным шумом обеих выборок:
    E[Σ Δp²/p̄] = Σ Δp_true²/p̄ + (N−1)(1/n₁+1/n₂)."""
    n1, n2 = c1.sum(-1), c2.sum(-1)
    N = c1.shape[-1]
    p1, p2 = c1 / n1[..., None], c2 / n2[..., None]
    pbar = (c1 + c2) / (n1 + n2)[..., None]
    s = ((p1 - p2) ** 2 / np.maximum(pbar, 1e-12)).sum(-1)
    return np.sqrt(np.maximum(0, s - (N - 1) * (1 / n1 + 1 / n2)))


def report(a):
    rng = np.random.default_rng(0)
    out_dir = Path(a.out or a.dirs[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    T, configs, meta = load(a.dirs)
    if a.variants:
        T, configs = select(T, configs, a.variants.split(","))
    N = meta["sides"]
    ro = a.readout
    face = T[ro]
    ok = face >= 0
    masses = list(dict.fromkeys(c["mass"] for c in configs))
    variants = list(dict.fromkeys(c["variant"] for c in configs))
    style = {m: MASS_STYLE[i % len(MASS_STYLE)] for i, m in enumerate(masses)}

    def counts_of(sel):
        return np.bincount(face[sel & ok], minlength=N).astype(float)

    lines = [f"# d{N} seed {meta['seed']}: {len(T)} бросков, readout = {ro}", ""]
    lines.append(f"Параметры броска: {json.dumps({k: meta['cfg'][k] for k in ('mu', 'e', 'roll', 'height', 'speed', 'spin')}, ensure_ascii=False)}")
    lines.append("")
    lines.append("RMS — среднеквадратичный перекос p_i·N от 1 с вычтенным шумом выборки; "
                 "квазистатика — то же для площадей Вороного; β — доля квазистатического перекоса, "
                 "дошедшая до динамики (1 = чистая квазистатика, 0 = динамика всё размыла).")
    lines.append("")
    lines.append("| конфигурация | бросков | на угол | χ² p-value | RMS, % [95% ДИ] | квазистатика RMS, % | max |p·N−1|, % | β [95% ДИ] |")
    lines.append("|---|---|---|---|---|---|---|---|")

    summary = []
    for ci, c in enumerate(configs):
        sel = T["shape"] == ci
        cnt = counts_of(sel)
        n = cnt.sum()
        E = n / N
        chi2 = ((cnt - E) ** 2 / E).sum()
        pval = chi2_dist.sf(chi2, N - 1)
        q = quasi_probs(c, ro)
        x = q * N - 1
        r = boot_rms(cnt, rng)
        b = boot(cnt, lambda s: beta(s, x), rng) if x @ x > 1e-6 else (np.nan,) * 3
        cocked = (T["bottom"][sel] < 0).mean()
        summary.append(dict(cfg=c, counts=cnt, x=x))
        lines.append(
            f"| {c['name']} | {int(sel.sum())} | {cocked:.1%} | {pval:.2g} | {100 * r[0]:.2f} [{100 * r[1]:.2f}, {100 * r[2]:.2f}] "
            f"| {100 * np.sqrt((x ** 2).mean()):.2f} | {100 * np.abs(cnt / n * N - 1).max():.1f} "
            f"| {b[0]:.2f} [{b[1]:.2f}, {b[2]:.2f}] |")
    if ro == "top":
        never = [c["name"] for c in configs if (quasi_probs(c, "top") == 0).any()]
        if never:
            lines += ["", f"⚠️ readout=top: у {', '.join(never)} есть грани, которые никогда не оказываются строго сверху "
                          "(нет противоположной грани) — для нечётных/несимметричных dN читайте нижнюю грань."]

    pairs = [(i, j) for i in range(len(configs)) for j in range(i + 1, len(configs))
             if configs[i]["mass"] == configs[j]["mass"] and configs[i]["flat"] == configs[j]["flat"]]
    if pairs:
        lines += ["", "## Попарное сравнение (одинаковая масса)", "",
                  "χ² — тест однородности двух распределений граней; RMS разницы — с вычтенным шумом обеих выборок.", "",
                  "| A | B | χ² p-value | RMS разницы, % [95% ДИ] | max |Δ(p·N)|, % (грань) |", "|---|---|---|---|---|"]
        for i, j in pairs:
            c1, c2 = summary[i]["counts"], summary[j]["counts"]
            pval = chi2_contingency(np.stack([c1, c2]))[1]
            sims1 = rng.multinomial(c1.sum(), c1 / c1.sum(), N_BOOT)
            sims2 = rng.multinomial(c2.sum(), c2 / c2.sum(), N_BOOT)
            est = float(rms_diff(c1[None], c2[None])[0])
            lo, med, hi = np.percentile(rms_diff(sims1, sims2), [2.5, 50, 97.5])
            d = (c1 / c1.sum() - c2 / c2.sum()) * N
            k = int(np.abs(d).argmax())
            lines.append(f"| {configs[i]['name']} | {configs[j]['name']} | {pval:.2g} "
                         f"| {100 * est:.2f} [{100 * max(lo - med + est, 0):.2f}, {100 * (hi - med + est):.2f}] "
                         f"| {100 * d[k]:+.1f} ({k + 1}) |")

    if len({c["flat"] for c in configs}) > 1:
        _fig_flat_sweep(T, configs, summary, N, style, out_dir / f"flat_sweep_{ro}.png", rng)
        lines += ["", f"График: flat_sweep_{ro}.png"]
    else:
        _fig_faces(summary, masses, variants, N, ro, out_dir / f"faces_{ro}.png")
        _fig_convergence(T, configs, face, ok, N, ro, style, out_dir / f"convergence_{ro}.png", rng)
        _fig_params(T, configs, face, ok, N, ro, style, a.bins, out_dir / f"params_{ro}.png", rng)
        _fig_heat(T, configs, face, ok, N, masses, variants, a.bins, out_dir / f"heat_mu_e_{ro}.png")
        _fig_dynamics(T, configs, style, a.bins, out_dir / "dynamics.png")
        lines += ["", f"Графики: faces_{ro}.png, convergence_{ro}.png, params_{ro}.png, heat_mu_e_{ro}.png, dynamics.png"]
    text = "\n".join(lines)
    (out_dir / f"report_{ro}.md").write_text(text + "\n")
    print(text)


def _fig_faces(summary, masses, variants, N, ro, fn):
    fig, axes = plt.subplots(len(masses), 1, figsize=(max(8, 0.55 * N + 3), 2.8 * len(masses)),
                             sharex=True, squeeze=False)
    offs = {v: (i - (len(variants) - 1) / 2) * 0.22 for i, v in enumerate(variants)}
    for ax, mass in zip(axes[:, 0], masses):
        for s in summary:
            c = s["cfg"]
            if c["mass"] != mass:
                continue
            cnt, n = s["counts"], s["counts"].sum()
            p = cnt / n
            se = np.sqrt(p * (1 - p) / n) * N
            xs = np.arange(1, N + 1) + offs[c["variant"]]
            col = VARIANT_COLOR.get(c["variant"], INK2)
            ax.errorbar(xs, p * N, yerr=1.96 * se, fmt="o", ms=5, color=col, capsize=0, lw=1.5,
                        label=f"{c['variant']}: симуляция")
            ax.plot(xs, s["x"] + 1, marker="_", ms=14, mew=2, ls="none", color=col, alpha=0.55,
                    label=f"{c['variant']}: квазистатика (Вороной)")
        ax.axhline(1, color=INK2, lw=1, ls="--")
        ax.set_title(f"масса: {mass}", loc="left", fontsize=10, color=INK2)
        ax.set_ylabel("p · N")
    axes[0, 0].legend(ncol=2, fontsize=8, frameon=False, loc="upper right")
    axes[-1, 0].set_xticks(range(1, N + 1))
    axes[-1, 0].set_xlabel(f"грань ({'на столе' if ro == 'bottom' else 'сверху'}), 95% ДИ")
    fig.suptitle(f"Вероятности граней d{N} относительно 1/N", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(fn, dpi=130)
    plt.close(fig)


def sphere_fraction(normals, flat, n=200_000):
    """Доля направлений из ЦМ, в которых луч упирается в сферический остаток, а не в грань."""
    u = np.random.default_rng(0).normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return float(((u @ normals.T).max(1) < flat).mean())


def _fig_flat_sweep(T, configs, summary, N, style, fn, rng):
    groups = {}
    for ci, c in enumerate(configs):
        groups.setdefault((c["variant"], c["mass"]), []).append(ci)
    fig = plt.figure(figsize=(13, 9.5), layout="constrained")
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.15])
    ax_rms, ax_geo = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_t, ax_max = fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 0])
    ax_face = fig.add_subplot(gs[2, :])
    for (variant, mass), idx in groups.items():
        idx = sorted(idx, key=lambda i: configs[i]["flat"])
        f = np.array([configs[i]["flat"] for i in idx])
        col = VARIANT_COLOR.get(variant, INK2)
        kw = dict(color=col, ls=style[mass], marker="o", ms=4, label=f"{variant}/{mass}")
        r = np.array([boot_rms(summary[i]["counts"], rng) for i in idx])
        ax_rms.plot(f, 100 * r[:, 0], **kw)
        ax_rms.fill_between(f, 100 * r[:, 1], 100 * r[:, 2], color=col, alpha=0.12, lw=0)
        mx = [100 * np.abs(summary[i]["counts"] / summary[i]["counts"].sum() * N - 1).max() for i in idx]
        ax_max.plot(f, mx, **kw)
        ck = [100 * (T["bottom"][T["shape"] == i] < 0).mean() for i in idx]
        ax_geo.plot(f, ck, **kw)
        ax_geo.plot(f, [100 * sphere_fraction(configs[i]["normals"], configs[i]["flat"]) for i in idx],
                    color=col, ls=":", marker="s", ms=3, label=f"{variant}: доля сферы в направлениях из ЦМ")
        ax_t.plot(f, [np.median(T["t"][T["shape"] == i]) for i in idx], **kw)
    # шум честного кубика для RMS при таком объёме: 95% верхняя граница оценки
    n_med = np.median([s["counts"].sum() for s in summary])
    ax_rms.axhline(100 * np.sqrt(chi2_dist.ppf(0.95, N - 1) - (N - 1)) / np.sqrt(n_med), color=INK2, lw=1, ls="--",
                   label=f"порог шума (95%) при n≈{n_med:.0f}")
    ax_rms.set_ylabel("RMS перекос граней, %")
    ax_max.set_ylabel("max |p·N − 1|, %")
    ax_geo.set_ylabel("%")
    ax_geo.set_title("замер на угол (сплошные) и доля сферы (точки)", fontsize=9, loc="left", color=INK2)
    ax_t.set_ylabel("медиана времени до покоя, с")
    for ax in (ax_rms, ax_geo, ax_t, ax_max):
        ax.set_xlabel("CUT_FACTOR")
        ax.legend(fontsize=8, frameon=False)

    # p·N по граням для первой группы
    (variant, mass), idx = next(iter(groups.items()))
    idx = sorted(idx, key=lambda i: configs[i]["flat"])
    M = np.array([summary[i]["counts"] / summary[i]["counts"].sum() * N for i in idx])
    lim = max(0.05, np.abs(M - 1).max())
    ax_face.grid(False)
    im = ax_face.imshow(M.T, cmap="RdBu_r", vmin=1 - lim, vmax=1 + lim, aspect="auto", origin="lower")
    for a_ in range(M.shape[0]):
        for b_ in range(N):
            ax_face.text(a_, b_, f"{M[a_, b_]:.2f}", ha="center", va="center", fontsize=6.5,
                         color="white" if abs(M[a_, b_] - 1) > 0.6 * lim else INK)
    ax_face.set_xticks(range(len(idx)), [f"{configs[i]['flat']:.3f}" for i in idx], fontsize=8)
    ax_face.set_yticks(range(N), range(1, N + 1), fontsize=8)
    ax_face.set_xlabel("CUT_FACTOR")
    ax_face.set_ylabel("грань")
    ax_face.set_title(f"{variant}/{mass}: p·N по граням", fontsize=9, loc="left", color=INK2)
    fig.colorbar(im, ax=ax_face, shrink=0.9, label="p·N")
    fig.suptitle(f"d{N}: честность в зависимости от CUT_FACTOR", x=0.01, ha="left")
    fig.savefig(fn, dpi=130)
    plt.close(fig)


def _fig_convergence(T, configs, face, ok, N, ro, style, fn, rng):
    C = len(configs)
    ncol = 2
    nrow = 1 + (C + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.6 * nrow), squeeze=False)
    ax_rms, ax_raw = axes[0]
    for ci, c in enumerate(configs):
        col = VARIANT_COLOR.get(c["variant"], INK2)
        # бросок в файле упорядочен по времени завершения внутри чанка — перемешиваем
        f = rng.permutation(face[(T["shape"] == ci) & ok])
        n_tot = len(f)
        ns = np.unique(np.geomspace(min(50, n_tot), n_tot, 40).astype(int))
        onehot = np.zeros((n_tot, N))
        onehot[np.arange(n_tot), f] = 1
        cum = np.cumsum(onehot, 0)[ns - 1]
        est = np.array([boot_rms(cnt, rng) for cnt in cum])
        raw = np.sqrt(((cum - ns[:, None] / N) ** 2 / (ns[:, None] / N)).sum(1) / ns)
        kw = dict(color=col, ls=style[c["mass"]])
        ax_rms.plot(ns, 100 * est[:, 0], label=c["name"], **kw)
        ax_rms.fill_between(ns, 100 * est[:, 1], 100 * est[:, 2], color=col, alpha=0.08, lw=0)
        ax_raw.plot(ns, 100 * raw, label=c["name"], **kw)

        ax = axes[1 + ci // ncol, ci % ncol]
        p = cum / ns[:, None] * N
        final = np.abs(p[-1] - 1)
        top = sorted(np.argsort(final)[-3:], key=lambda k: p[-1, k])
        label_y = [p[-1, k] for k in top]
        for m in range(1, len(label_y)):
            label_y[m] = max(label_y[m], label_y[m - 1] + 0.05)
        for k in range(N):
            if k not in top:
                ax.plot(ns, p[:, k], color=INK2, lw=0.8, alpha=0.35)
        for k, ly in zip(top, label_y):
            ax.plot(ns, p[:, k], color=col, lw=1.8)
            ax.annotate(str(k + 1), (ns[-1], ly), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK2, annotation_clip=False)
        band = 1.96 * np.sqrt((1 / N) * (1 - 1 / N) / ns) * N
        ax.fill_between(ns, 1 - band, 1 + band, color=GRID, alpha=0.6, lw=0, label="95% шума у честного кубика")
        ax.set_xscale("log")
        ax.set_ylim(0.4, 1.6)
        ax.set_title(f"{c['name']}: p·N по граням (3 самых отклонившихся подписаны)", fontsize=9, loc="left", color=INK2)
        ax.set_xlabel("бросков")
    for ax in axes[0]:
        ax.set_xscale("log")
        ax.set_xlabel("бросков данной конфигурации")
    n_ref = np.geomspace(50, max(np.bincount(T["shape"])), 50)
    ax_raw.plot(n_ref, 100 * np.sqrt((N - 1) / n_ref), color=INK2, lw=1, ls="--", label="шум честного кубика √((N−1)/n)")
    ax_rms.set_ylabel("RMS перекос (без шума), %")
    ax_raw.set_ylabel("RMS отклонение как есть, %")
    ax_raw.set_yscale("log")
    ax_rms.legend(fontsize=8, frameon=False)
    ax_raw.legend(fontsize=8, frameon=False)
    axes[1, 0].legend(fontsize=8, frameon=False, loc="lower right")
    for k in range(C, (nrow - 1) * ncol):
        axes[1 + k // ncol, k % ncol].axis("off")
    fig.suptitle(f"d{N}: сходимость статистики по числу бросков", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(fn, dpi=130)
    plt.close(fig)


def _fig_params(T, configs, face, ok, N, ro, style, k, fn, rng):
    params = [("mu", "трение μ"), ("e", "коэффициент отскока e"), ("roll", "трение качения, м"),
              ("v", "скорость броска, м/с"), ("w", "угловая скорость, рад/с")]
    params = [p for p in params if len(np.unique(T[p[0]])) > 1]
    fig, axes = plt.subplots(2, len(params), figsize=(4.2 * len(params), 7), squeeze=False)
    for j, (key, label) in enumerate(params):
        centers, idx, _ = _bins(T[key], k)
        for ci, c in enumerate(configs):
            col = VARIANT_COLOR.get(c["variant"], INK2)
            x = quasi_probs(c, ro) * N - 1
            rows = []
            for b in range(len(centers)):
                sel = (T["shape"] == ci) & (idx == b) & ok
                cnt = np.bincount(face[sel], minlength=N).astype(float)
                r = boot_rms(cnt, rng)
                bb = boot(cnt, lambda s: beta(s, x), rng) if x @ x > 1e-6 else (np.nan,) * 3
                rows.append((*r, *bb))
            rows = np.array(rows)
            kw = dict(color=col, ls=style[c["mass"]], marker="o", ms=4)
            axes[0, j].plot(centers, 100 * rows[:, 0], label=c["name"], **kw)
            axes[0, j].fill_between(centers, 100 * rows[:, 1], 100 * rows[:, 2], color=col, alpha=0.08, lw=0)
            if not np.isnan(rows[:, 3]).all():
                axes[1, j].plot(centers, rows[:, 3], label=c["name"], **kw)
                axes[1, j].fill_between(centers, rows[:, 4], rows[:, 5], color=col, alpha=0.08, lw=0)
        axes[0, j].set_xlabel(label)
        axes[1, j].set_xlabel(label)
        axes[1, j].axhline(1, color=INK2, lw=1, ls="--")
        axes[1, j].axhline(0, color=INK2, lw=1)
    axes[0, 0].set_ylabel("RMS перекос граней, %")
    axes[1, 0].set_ylabel("β: доля квазистатического перекоса")
    axes[0, -1].legend(fontsize=8, frameon=False)
    fig.suptitle(f"d{N}: перекос в зависимости от параметров (бины по квантилям, тень — 95% ДИ)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(fn, dpi=130)
    plt.close(fig)


def _fig_heat(T, configs, face, ok, N, masses, variants, k, fn):
    _, im_, mu_lab = _bins(T["mu"], k)
    _, ie_, e_lab = _bins(T["e"], k)
    nm, ne = len(mu_lab), len(e_lab)
    grids = {}
    for ci, c in enumerate(configs):
        g = np.full((ne, nm), np.nan)
        for i in range(ne):
            for j in range(nm):
                sel = (T["shape"] == ci) & (ie_ == i) & (im_ == j) & ok
                if sel.sum() > 5 * N:
                    g[i, j] = 100 * rms_bias(np.bincount(face[sel], minlength=N))[0]
        grids[c["name"]] = g
    vmax = max([np.nanmax(g) for g in grids.values() if not np.isnan(g).all()] + [1e-9])
    fig, axes = plt.subplots(len(variants), len(masses), figsize=(3.6 * len(masses) + 1, 3.4 * len(variants)),
                             squeeze=False, layout="constrained")
    for (vi, v), (mi, mass) in ((x, y) for x in enumerate(variants) for y in enumerate(masses)):
        ax = axes[vi, mi]
        g = grids.get(f"{v}/{mass}")
        ax.grid(False)
        if g is None:
            ax.axis("off")
            continue
        ax.imshow(g, origin="lower", cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        for i in range(ne):
            for j in range(nm):
                if not np.isnan(g[i, j]):
                    ax.text(j, i, f"{g[i, j]:.1f}", ha="center", va="center", fontsize=8,
                            color="white" if g[i, j] > 0.6 * vmax else INK)
        ax.set_xticks(range(nm), mu_lab, fontsize=7, rotation=30)
        ax.set_yticks(range(ne), e_lab, fontsize=7)
        ax.set_title(f"{v} / {mass}", fontsize=9, color=INK2)
        if vi == len(variants) - 1:
            ax.set_xlabel("μ")
        if mi == 0:
            ax.set_ylabel("e")
    fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(0, vmax), "Blues"), ax=axes, shrink=0.8, label="RMS перекос, %")
    fig.suptitle(f"d{N}: RMS перекос граней по сетке μ × e", x=0.01, ha="left")
    fig.savefig(fn, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _fig_dynamics(T, configs, style, k, fn):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for j, (key, label) in enumerate([("mu", "трение μ"), ("e", "коэффициент отскока e")]):
        centers, idx, _ = _bins(T[key], k)
        for ci, c in enumerate(configs):
            col = VARIANT_COLOR.get(c["variant"], INK2)
            sel = T["shape"] == ci
            tm = [np.median(T["t"][sel & (idx == b)]) for b in range(len(centers))]
            ck = [100 * (T["bottom"][sel & (idx == b)] < 0).mean() for b in range(len(centers))]
            kw = dict(color=col, ls=style[c["mass"]], marker="o", ms=4, label=c["name"])
            axes[0, j].plot(centers, tm, **kw)
            axes[1, j].plot(centers, ck, **kw)
        axes[0, j].set_xlabel(label)
        axes[1, j].set_xlabel(label)
    axes[0, 0].set_ylabel("медиана времени до покоя, с")
    axes[1, 0].set_ylabel("замер на угол (не на грани), %")
    axes[0, 1].legend(fontsize=8, frameon=False)
    fig.suptitle("Динамика бросков", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(fn, dpi=130)
    plt.close(fig)
