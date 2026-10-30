# -*- coding: utf-8 -*-
"""Fig. S4 失配迁移的 16 格地图：BCREA (band) 对五个对手的整池 RC 胜负（v3 批次）。

图级主张：BCREA 的鲁棒优势有明确边界——对名义池几乎全格赢且优势随扰动强度单调增大；
对 sampled worst-case 16 格全输（小幅但系统性）；对 pessimistic 只在 Γ=1 的轻预算档占优、
在它自己的训练角点 Γ=C 被反超；对 NSGA-II 互有胜负。
数据契约：20261013_v3_analysis/v3_grid.csv 的 rc_full_pool，按 (instance, seed, cell) 配对；
每格 n = 30 实例 × 10 种子 = 300 对，与 SI Table（失配 16 格表）同源同值。
画法：颜色 = band 严格高于对手的单元比例（0–100%，50% 为五五开）；格内数字 = 配对差
median(RC_band − RC_opponent)（kills），正 = BCREA 更好；训练档 d30_g2 实线加框，
Γ=C 列（单场景角点）刻度标签置灰。显著性检验在 SI Wilcoxon 表（配对 + Holm）。
运行：py -3.11 code/figure/figS4_mismatch.py
"""
import style
style.apply()
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from audit_panel_alignment import require_matplotlib_panel_alignment

io = style.load_io()
C = style.OI

DELTAS = (0.1, 0.2, 0.3, 0.4)
GAMMAS = (1, 2, 3, 12)
TRAIN_CELL = (2, 1)
OPPONENTS = [("nominal", "vs Nominal"), ("pessimistic", "vs Pessimistic"),
             ("sampled_wc", "vs Sampled WC"), ("nsga2", "vs NSGA-II"),
             ("bcrea_strict", "vs BCREA strict")]

DIV = LinearSegmentedColormap.from_list("wl_div", [C["vermillion"], "#f7f7f7", C["blue"]])


def load_grid() -> tuple[dict, int]:
    g = io.load_analysis("v3_grid")
    w = g.pivot_table(index=["instance", "seed", "cell"], columns="method",
                      values="rc_full_pool").reset_index()
    return w, len(w) // (len(DELTAS) * len(GAMMAS))


def stats(w, opponent: str):
    win = np.zeros((len(DELTAS), len(GAMMAS)))
    med = np.zeros_like(win)
    d = w["bcrea_band"] - w[opponent]
    for i, dl in enumerate(DELTAS):
        for j, gm in enumerate(GAMMAS):
            cell = f"d{int(round(dl * 100)):02d}_g{gm}"
            s = d[w.cell == cell]
            win[i, j] = float(np.mean(s > 1e-9))
            med[i, j] = float(np.median(s))
    return win, med


def panel(ax, win, med, title, first: bool):
    im = ax.imshow(win, cmap=DIV, norm=TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0),
                   aspect="auto")
    for i in range(win.shape[0]):
        for j in range(win.shape[1]):
            hot = abs(win[i, j] - 0.5) > 0.28
            col = "white" if hot else "#222222"
            val = 0.0 if abs(med[i, j]) < 0.05 else med[i, j]      # 避免 "-0.0"
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                    fontsize=8, color=col, fontweight="bold")
    row, col = TRAIN_CELL
    for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        x0, y0 = col + dx * 0.5, row + dy * 0.5
        ax.plot([x0, x0 - dx * 0.18], [y0, y0], color="#000000", lw=1.4, zorder=5,
                solid_capstyle="butt")
        ax.plot([x0, x0], [y0, y0 - dy * 0.18], color="#000000", lw=1.4, zorder=5,
                solid_capstyle="butt")
    ax.set_xticks(range(len(GAMMAS)), ["1", "2", "3", "C"])
    ax.get_xticklabels()[-1].set_color("#777777")
    if first:
        ax.set_yticks(range(len(DELTAS)), [f"{d:.1f}" for d in DELTAS])
        ax.set_ylabel(r"degradation $\delta$")
    else:
        ax.set_yticks([])
    ax.set_xlabel(r"budget $\Gamma$")
    ax.set_title(title, fontsize=9, pad=5)
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def main():
    style.require_batch()
    style.QA_DIR.mkdir(parents=True, exist_ok=True)
    w, n_pairs = load_grid()

    fig = plt.figure(figsize=(170 * style.MM, 50 * style.MM))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.06],
                          left=0.055, right=0.98, bottom=0.22, top=0.86, wspace=0.22)
    im = None
    for k, (opp, title) in enumerate(OPPONENTS):
        ax = fig.add_subplot(gs[0, k])
        win, med = stats(w, opp)
        im = panel(ax, win, med, title, first=(k == 0))
        style.panel_tag(ax, f"({chr(ord('a') + k)})", dx=-0.02, dy=1.16)
    cax = fig.add_subplot(gs[0, 5])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("share of units\nBCREA RC $>$ opponent (%)", fontsize=8)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(["0", "25", "50", "75", "100"])
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_linewidth(0.4)

    require_matplotlib_panel_alignment(
        fig, json_out=str(style.QA_DIR / "figS4_mismatch.alignment.json"),
        overlay_svg=str(style.QA_DIR / "figS4_mismatch.alignment.svg"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=False, exclude_axes=(cax,))
    style.save(fig, "figS4_mismatch")

    for opp, title in OPPONENTS:
        win, med = stats(w, opp)
        print(f"[figS4] {title:18s} win {win.min() * 100:5.1f}–{win.max() * 100:5.1f}%  "
              f"median dRC {med.min():+.2f} … {med.max():+.2f}")
    print(f"[figS4] pairs per cell = {n_pairs}; training cell d30_g2 outlined")


if __name__ == "__main__":
    main()
