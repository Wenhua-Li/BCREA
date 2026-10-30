# -*- coding: utf-8 -*-
"""Fig.6 机制与失配边界：带内选择在同预算下压低带内退化，优势边界明确。

图级主张：BCREA 的带内选择把"带内退化"在预算内压到接近 0，而名义目标只让出很小代价；
这种优势在 δ=0.1、Γ=1 的轻扰动格子上消失（诚实呈现弱格）。
数据契约（全部来自 v3 批次的分析层，无消融批次）：
- (a) 收敛 trace：20261013_v3_analysis/v3_traces.csv（50 个预算分位点，按方法×规模中位/IQR）。
  左轴 = 名义目标（kills）：BCREA band 与 nominal 两条；右轴 = 带内退化（kills）：BCREA band。
  **口径**：band 的退化是训练口径（自家 K=8 分层场景），不是 66 角点评估，禁止跨方法比绝对值。
- (b) 16 格失配边界：v3_grid.csv 的整池 RC（rc_full_pool），按 (instance, seed, cell) 配对
  算 band − nominal 的符号与差值；格内 = 配对胜率与中位 ΔRC。训练档 d30_g2 实线加框；
  弱格 d10_g1（胜率 43.7%）用 vermillion 虚线框标出；Γ=C 列（单场景）灰斜纹标注。
  vs pessimistic 的边界留 SI。
- n：(a) 每规模 160/70/70 个 run（面板标题外，见打印）；(b) 每格 30 实例 × 10 种子 = 300 对。
统计：中位数+IQR；图注沿用脚本打印的计数。
运行：py -3.11 code/figure/fig6_mechanism.py
"""
import style
style.apply()
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle

from audit_panel_alignment import require_matplotlib_panel_alignment

io = style.load_io()
C = style.OI

DELTAS = (0.1, 0.2, 0.3, 0.4)
GAMMAS = (1, 2, 3, 12)
TRAIN_CELL = (2, 1)                 # d30_g2 在 (δ,Γ) 网格里的位置
DEG_MAX = 7.2                       # 右轴上限（三面板共用，便于跨规模读）

# 胜率发散色：vermillion（输）→ 白（五五开）→ 蓝（BCREA 赢），色盲可分
DIV = LinearSegmentedColormap.from_list(
    "wl_div", [C["vermillion"], "#f7f7f7", C["blue"]])


def load_traces() -> dict:
    t = io.load_analysis("v3_traces")
    out = {}
    for n in io.SCALES:
        s = t[t.n_target == n]
        out[n] = {
            "band_nom": s[(s.method == "bcrea_band") & (s.series == "nominal_obj")]
            .sort_values("budget_frac"),
            "band_deg": s[(s.method == "bcrea_band") & (s.series == "in_band_deg")]
            .sort_values("budget_frac"),
            "nom_nom": s[(s.method == "nominal") & (s.series == "nominal_obj")]
            .sort_values("budget_frac"),
        }
    return out


def load_grid() -> tuple[np.ndarray, np.ndarray, int]:
    g = io.load_analysis("v3_grid")
    w = g.pivot_table(index=["instance", "seed", "cell"], columns="method",
                      values="rc_full_pool").reset_index()
    w["d"] = w["bcrea_band"] - w["nominal"]
    win = np.zeros((len(DELTAS), len(GAMMAS)))
    med = np.zeros_like(win)
    for i, d in enumerate(DELTAS):
        for j, gm in enumerate(GAMMAS):
            cell = f"d{int(round(d * 100)):02d}_g{gm}"
            s = w[w.cell == cell]
            win[i, j] = float(np.mean(s.d > 1e-9))
            med[i, j] = float(np.median(s.d))
    return win, med, len(w)


def panel_trace(ax, d: dict, n_target: int, show_right_label: bool):
    x = d["band_nom"].budget_frac.to_numpy() * 100.0
    band = d["band_nom"]
    nom = d["nom_nom"]
    deg = d["band_deg"]

    ax.fill_between(x, band.q25, band.q75, color=style.METHOD["bcrea_band"]["color"],
                    alpha=0.12, lw=0)
    ax.plot(nom.budget_frac * 100, nom["median"], color=style.METHOD["nominal_kcea"]["color"],
            lw=style.LW_EMPH, alpha=0.55, zorder=2)
    ax.plot(x, band["median"], color=style.METHOD["bcrea_band"]["color"],
            lw=style.LW_DATA, zorder=3)

    ax2 = ax.twinx()
    ax2.fill_between(deg.budget_frac * 100, deg.q25, deg.q75,
                     color=style.METHOD["bcrea_band"]["color"], alpha=0.10, lw=0)
    ax2.plot(deg.budget_frac * 100, deg["median"], color=style.METHOD["bcrea_band"]["color"],
             ls="--", lw=style.LW_DATA, zorder=3)
    ax2.set_ylim(0, DEG_MAX)
    ax2.set_yticks([0, 2, 4, 6])
    if show_right_label:
        ax2.set_ylabel("in-band loss, training (targets)", fontsize=8)

    ys = np.concatenate([band["median"], nom["median"]])
    pad = 0.04 * float(ys.max() - ys.min())
    ax.set_ylim(float(ys.min()) - pad, float(ys.max()) + pad)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("decode budget used (%)")
    ax.set_title(f"{n_target} targets", fontsize=9, pad=11)


def panel_grid(ax, win: np.ndarray, med: np.ndarray):
    im = ax.imshow(win, cmap=DIV, norm=TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0),
                   aspect="auto")
    for i in range(win.shape[0]):
        for j in range(win.shape[1]):
            hot = abs(win[i, j] - 0.5) > 0.28
            col = "white" if hot else "#222222"
            ax.text(j, i - 0.16, f"{med[i, j]:+.1f}", ha="center", va="center",
                    fontsize=8, color=col, fontweight="bold")
            ax.text(j, i + 0.22, f"{win[i, j] * 100:.0f}%", ha="center", va="center",
                    fontsize=8, color=col)
    row, col = TRAIN_CELL
    ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, ec="#000000",
                           lw=1.4, zorder=5))
    ax.add_patch(Rectangle((2.5, -0.5), 1.0, len(DELTAS), fill=False, ec="#888888",
                           hatch="////", lw=0.0, zorder=2))
    ax.add_patch(Rectangle((-0.5, -0.5), 1, 1, fill=False, ec=C["vermillion"], lw=1.0,
                           ls=(0, (2.2, 1.6)), zorder=5))
    ax.set_xticks(range(len(GAMMAS)), ["1", "2", "3", "C"])
    ax.get_xticklabels()[-1].set_color("#777777")
    ax.set_yticks(range(len(DELTAS)), [f"{d:.1f}" for d in DELTAS])
    ax.set_xlabel("budget $\\Gamma$")
    ax.set_ylabel("degradation $\\delta$")
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def main():
    style.require_batch()
    style.QA_DIR.mkdir(parents=True, exist_ok=True)
    tr = load_traces()
    win, med, n_pairs = load_grid()

    fig = plt.figure(figsize=(style.DOUBLE_COL, 68 * style.MM))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.45],
                          left=0.055, right=0.94, bottom=0.30, top=0.88,
                          wspace=0.72)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_g = fig.add_subplot(gs[0, 3])
    for i, (ax, n) in enumerate(zip(axes, io.SCALES)):
        panel_trace(ax, tr[n], n, show_right_label=(i == 2))
        style.panel_tag(ax, f"({chr(ord('a') + i)})", dx=-0.02, dy=1.08)
        if i == 0:
            ax.set_ylabel("nominal objective (targets)")
    im = panel_grid(ax_g, win, med)
    style.panel_tag(ax_g, "(d)", dx=-0.02, dy=1.08)

    cb = fig.colorbar(im, ax=ax_g, fraction=0.055, pad=0.03)
    cb.set_label("share of paired runs\nBCREA pool RC $>$ nominal", fontsize=8)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_linewidth(0.4)

    handles = [
        plt.Line2D([], [], color=style.METHOD["bcrea_band"]["color"], lw=style.LW_DATA,
                   label="BCREA (band): nominal objective"),
        plt.Line2D([], [], color=style.METHOD["nominal_kcea"]["color"], lw=style.LW_EMPH,
                   alpha=0.55, label="Nominal KCEA: nominal objective"),
        plt.Line2D([], [], color=style.METHOD["bcrea_band"]["color"], lw=style.LW_DATA,
                   ls="--", label="BCREA (band): in-band loss (training)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.005), handlelength=2.0, columnspacing=1.8)

    require_matplotlib_panel_alignment(
        fig, json_out=str(style.QA_DIR / "fig6_mechanism.alignment.json"),
        overlay_svg=str(style.QA_DIR / "fig6_mechanism.alignment.svg"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=False, exclude_axes=(cb.ax,))
    style.save(fig, "fig6_mechanism")

    for n in io.SCALES:
        d = tr[n]
        print(f"[fig6] (a) N={n}: nominal obj {d['band_nom']['median'].iloc[0]:.2f} -> "
              f"{d['band_nom']['median'].iloc[-1]:.2f} (BCREA) vs "
              f"{d['nom_nom']['median'].iloc[-1]:.2f} (nominal); in-band loss "
              f"{d['band_deg']['median'].iloc[0]:.2f} -> {d['band_deg']['median'].iloc[-1]:.3f}; "
              f"runs={int(d['band_nom'].n_runs.iloc[0])}")
    print(f"[fig6] (b) pairs={n_pairs}, overall win={np.mean(win):.3f}")
    for i, dl in enumerate(DELTAS):
        print("   delta=%.1f  " % dl + "  ".join(
            f"G{gm}: {med[i, j]:+.2f}/{win[i, j] * 100:.0f}%" for j, gm in enumerate(GAMMAS)))


if __name__ == "__main__":
    main()
