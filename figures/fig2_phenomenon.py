# -*- coding: utf-8 -*-
"""Fig.2 现象图：名义等价 ≠ 鲁棒等价（v3 正式批次）。

图级主张：名义目标上等价的方案池在预算化降级下系统性分化，名义排名对最坏情形近乎无信息。
数据契约（v3 scans 任务，每实例每方法每种子一份 JSON；**不是 npz**）：
- 输入：style.require_batch()/"scans"/<method_prefix>_<instance>_s<seed>.json，
  前缀换算见 io.METHOD_PREFIX；主工作点 cells.d30_g2（δ=0.3、Γ=2，66 模式）：
  objsS（66×200 方案×模式质量）、deg（逐方案最坏退化）、nominal.eq_idx（等价带成员下标）。
- 面板：
  (a) gen100_conc_01 / s01 / nominal 池：66 模式 × 16 个等价带方案热图
      （方案按最坏值降序等距抽取，端点即全局最优/最差，不做挑选）；模式按展示方案上的
      均值排序（良性 → 严酷）；
  (b) 三规模 nominal 池带内成员的逐方案 Δ=objs0−min_s f_s(x) 分布（violin：
      160/70/70 个 run、7,680/6,720/14,000 个方案值；轴旁 n=run 数）；
      名义全并列比例取 20261013_v3_analysis/v3_phenomenon_scale.csv 的 share_nominal_all_tied；
  (c) 名义池整池 RC 与最优单解 WC 的差距 rc_full_pool − best_single_wc（逐 run 点 + 中位杠
      + IQR），源 20261013_v3_analysis/v3_rc.csv（method=nominal）。
- 统计：中位数+IQR；抖动用黄金角确定性序列（重跑逐点一致，不用随机数）；
  n 标在轴旁，图注沿用脚本打印的计数。
运行：py -3.11 code/figure/fig2_phenomenon.py
"""
import style
style.apply()
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from audit_panel_alignment import require_matplotlib_panel_alignment

io = style.load_io()
C = style.OI

PANEL_INSTANCE = "gen100_conc_01"      # 100 目标代表实例（固定规则，见 docstring）
PANEL_SEED = 1
PANEL_METHOD = "nominal_kcea"
CELL = "d30_g2"
N_SHOW = 16                            # 展示方案数（按最坏值等距抽取）

# 蓝系单色渐变（灰度/黑白打印可读），与 style 的蓝族同色相
BLUES = LinearSegmentedColormap.from_list(
    "blues6", ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"])


def load_panel_a() -> dict:
    arr = io.scan_arrays(io.load_scan(PANEL_METHOD, PANEL_INSTANCE, PANEL_SEED), CELL)
    M = arr["objsS"][:, arr["eq_idx"]]
    wc = M.min(axis=0)
    order = np.argsort(wc)[::-1]                       # 方案：鲁棒 → 脆
    sel = order[np.linspace(0, len(order) - 1, N_SHOW).round().astype(int)]
    modes = np.argsort(-M[:, sel].mean(axis=1))        # 模式：良性 → 严酷（沿模式轴排）
    shown = M[np.ix_(modes, sel)]
    assert wc[sel][0] == wc[sel].max() and wc[sel][-1] == wc[sel].min(), \
        "展示方案的首/末行必须是全局最鲁棒/最脆（括号端点数字才成立）"
    return {
        "matrix": shown,
        "wc_shown": wc[sel],
        "wc_all": wc,
        "eq_size": int(len(arr["eq_idx"])),
        "pool_size": int(arr["objs0"].size),
        "nominal": float(arr["best"]),
        "span": float(wc.max() - wc.min()),
    }


def load_panel_b() -> dict:
    phen = io.load_analysis("v3_phenomenon_scale").set_index("n_target")
    out = {}
    for n in io.SCALES:
        vals, n_run = [], 0
        for inst in io.instances(n):
            for seed in io.SEEDS:
                arr = io.scan_arrays(io.load_scan("nominal_kcea", inst, seed), CELL)
                vals.append(arr["deg"][arr["eq_idx"]])
                n_run += 1
        v = np.concatenate(vals)
        out[n] = {
            "deg": v, "n_run": n_run, "n_plan": int(v.size),
            "tied": float(phen.loc[n, "share_nominal_all_tied"]),
            "median": float(np.median(v)),
        }
    return out


def load_panel_c() -> dict:
    rc = io.load_analysis("v3_rc")
    nom = rc[rc.method == "nominal"].copy()
    nom["gap"] = nom.rc_full_pool - nom.best_single_wc
    return {n: nom.loc[nom.n_target == n, "gap"].to_numpy(float) for n in io.SCALES}


def panel_heatmap(ax, fig, d: dict):
    im = ax.imshow(d["matrix"].T, aspect="auto", cmap=BLUES, interpolation="nearest",
                   vmin=float(d["matrix"].min()), vmax=float(d["nominal"]), origin="upper")
    ax.set_xticks([0, d["matrix"].shape[0] - 1])
    ax.set_xticklabels(["benign", "severe"])
    ax.set_yticks([0, N_SHOW - 1])
    ax.set_yticklabels(["robust", "fragile"])
    ax.set_xlabel("degradation mode $u$")
    ax.set_ylabel("plans (nominally equal)")
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 右侧：展示方案最坏值的跨度（端点按构造就是全局最优/最差）
    x_br = d["matrix"].shape[0] - 0.6
    ax.set_xlim(-0.5, x_br + 8.0)
    top, bot = float(d["wc_shown"].max()), float(d["wc_shown"].min())
    ax.plot([x_br, x_br], [0, N_SHOW - 1], color="#333333", lw=0.7)
    for y in (0, N_SHOW - 1):
        ax.plot([x_br - 0.6, x_br + 0.6], [y, y], color="#333333", lw=0.7)
    # 注意 imshow(origin="upper") 的 y 轴是反的：y=0 在顶部
    ax.text(x_br + 1.0, 0, f"{top:.1f}", fontsize=8, va="center", ha="left")
    ax.text(x_br + 1.0, N_SHOW - 1, f"{bot:.1f}", fontsize=8, va="center", ha="left")
    ax.text(x_br + 1.0, (N_SHOW - 1) / 2, f"$\\Delta${top - bot:.1f}", fontsize=8,
            va="center", ha="left", color=C["vermillion"])

    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.07, aspect=18)
    cb.set_label("$f_u(x)$", fontsize=8)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_linewidth(0.4)
    style.panel_tag(ax, "(a)", dx=-0.13, dy=1.02)


def panel_violin(ax, d: dict):
    positions = np.arange(len(io.SCALES))
    data = [d[n]["deg"] for n in io.SCALES]
    colors = [style.SCALE[n]["color"] for n in io.SCALES]
    parts = ax.violinplot(data, positions=positions, widths=0.72,
                          showextrema=False, showmedians=False)
    for body, col in zip(parts["bodies"], colors):
        body.set_facecolor(col)
        body.set_edgecolor(col)
        body.set_alpha(0.30)
        body.set_linewidth(0.8)
    for i, (v, col) in enumerate(zip(data, colors)):
        q25, q50, q75 = np.quantile(v, [0.25, 0.5, 0.75])
        ax.plot([i, i], [q25, q75], color=col, lw=2.6, solid_capstyle="butt", zorder=3)
        ax.plot(i, q50, marker="o", ms=4.0, mfc="white", mec=col, mew=1.0, zorder=4)
    ax.text(0.5, 0.985, "tied: " + "/".join(
        f"{d[n]['tied'] * 100:.1f}".rstrip("0").rstrip(".") for n in io.SCALES) + "%",
        transform=ax.transAxes, fontsize=8, ha="center", va="top")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{n}\n$n$={d[n]['n_run']}" for n in io.SCALES])
    ax.set_xlim(-0.6, len(io.SCALES) - 0.4)
    ax.set_ylim(0, max(float(np.max(v)) for v in data) * 1.35)
    ax.set_xlabel("targets $N$")
    ax.set_ylabel("per-plan loss $\\Delta$ (targets)")
    style.panel_tag(ax, "(b)")


def panel_gap(ax, d: dict):
    positions = np.arange(len(io.SCALES))
    medians = []
    for i, n in enumerate(io.SCALES):
        g = d[n]
        col = style.SCALE[n]["color"]
        # 确定性抖动（黄金角序列）：重跑逐点一致，不引入随机数
        x = i + 0.16 * np.sin(np.arange(g.size) * 2.39996323)
        ax.plot(x, g, ls="", marker="o", ms=2.0, mfc=col, mec="none", alpha=0.55)
        q25, q50, q75 = np.quantile(g, [0.25, 0.5, 0.75])
        medians.append(q50)
        ax.errorbar([i], [q50], yerr=[[q50 - q25], [q75 - q50]], fmt="none",
                    ecolor="#333333", elinewidth=0.8, capsize=1.8, capthick=0.8, zorder=3)
        ax.plot([i - 0.30, i + 0.30], [q50, q50], color="#000000", lw=2.2,
                solid_capstyle="butt", zorder=4)
    ax.axhline(0.0, color=C["grey"], lw=style.LW_REF, ls="--", zorder=1)
    ax.text(0.5, 0.985, "med: " + "/".join(f"{m:.2f}" for m in medians),
            transform=ax.transAxes, fontsize=8, ha="center", va="top")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{n}\n$n$={d[n].size}" for n in io.SCALES])
    ax.set_xlim(-0.6, len(io.SCALES) - 0.4)
    ax.set_ylim(-0.05, max(float(np.max(d[n])) for n in io.SCALES) * 1.35)
    ax.set_xlabel("targets $N$")
    ax.set_ylabel("pool RC $-$ best WC (targets)")
    style.panel_tag(ax, "(c)")


def main():
    style.require_batch()
    style.QA_DIR.mkdir(parents=True, exist_ok=True)
    a = load_panel_a()
    b = load_panel_b()
    c = load_panel_c()

    fig = plt.figure(figsize=(style.DOUBLE_COL, 68 * style.MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.85, 1.0, 1.0],
                          left=0.075, right=0.99, bottom=0.25, top=0.93, wspace=0.52)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    panel_heatmap(ax_a, fig, a)
    panel_violin(ax_b, b)
    panel_gap(ax_c, c)

    require_matplotlib_panel_alignment(
        fig, json_out=str(style.QA_DIR / "fig2_phenomenon.alignment.json"),
        overlay_svg=str(style.QA_DIR / "fig2_phenomenon.alignment.svg"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=False)
    style.save(fig, "fig2_phenomenon")

    print(f"[fig2] (a) {PANEL_INSTANCE} s{PANEL_SEED:02d}: "
          f"{a['eq_size']}/{a['pool_size']} plans in band, nominal {a['nominal']:.2f}, "
          f"worst-case {float(a['wc_all'].min()):.2f}-{float(a['wc_all'].max()):.2f} "
          f"(span {a['span']:.2f})")
    for n in io.SCALES:
        print(f"[fig2] (b) N={n}: runs={b[n]['n_run']} plans={b[n]['n_plan']} "
              f"tied={b[n]['tied'] * 100:.1f}% median={b[n]['median']:.2f} "
              f"| (c) median gap={np.median(c[n]):.2f} "
              f"share>0={np.mean(c[n] > 1e-9) * 100:.1f}%")


if __name__ == "__main__":
    main()
