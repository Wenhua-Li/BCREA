# -*- coding: utf-8 -*-
"""Fig. S1 名义等价带的相对极差在 16 格 (δ,Γ) 网格上的分布（v3 批次）。

图级主张：名义等价带内的最坏目标极差随扰动强度 δ 与预算 Γ 单调增大；
主工作点 d30_g2 的三规模中位数 4.3 / 3.2 / 9.7 %（与正文 Table III 同源同值）。
数据契约：
- 20261013_v3_analysis/v3_grid.csv（28800 行 = 16 格 × 30 实例 × 10 种子 × 6 方法）；
  只取 method=nominal 的 rel_spread（= 等价带内最坏目标极差 / 名义最优，见 core.robust_metrics.cell_metrics）。
- 每格取跨单元中位数（单元 = 实例 × 种子，n = 160/70/70），与 Table III 的口径一致；
  旧版用均值且 vmax=12 截断 100 目标的最大格，本版全局共享色标、不截断。
运行：py -3.11 code/figure/figS1_grid.py
"""
import style
style.apply()
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from audit_panel_alignment import require_matplotlib_panel_alignment

io = style.load_io()

DELTAS = (0.1, 0.2, 0.3, 0.4)
GAMMAS = (1, 2, 3, 12)
TRAIN_CELL = (2, 1)                     # d30_g2


def load_grid() -> dict[int, np.ndarray]:
    """method=nominal 的 rel_spread → 每规模 4×4 中位数矩阵（%）。"""
    g = io.load_analysis("v3_grid")
    g = g[g.method == "nominal"]
    out = {}
    for n in io.SCALES:
        M = np.full((len(DELTAS), len(GAMMAS)), np.nan)
        s = g[g.n_target == n]
        for i, d in enumerate(DELTAS):
            for j, gm in enumerate(GAMMAS):
                cell = f"d{int(round(d * 100)):02d}_g{gm}"
                M[i, j] = float(np.median(s[s.cell == cell].rel_spread.to_numpy())) * 100.0
        out[n] = M
    return out


def main():
    style.require_batch()
    style.QA_DIR.mkdir(parents=True, exist_ok=True)
    mats = load_grid()
    vmax = max(float(np.nanmax(M)) for M in mats.values()) * 1.001   # 全局色标，不截断

    fig, axs = plt.subplots(1, 3, figsize=(style.DOUBLE_COL, 52 * style.MM))
    im = None
    for k, (ax, n) in enumerate(zip(axs, io.SCALES)):
        M = mats[n]
        im = ax.imshow(M, cmap="Blues", vmin=0.0, vmax=vmax, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                hot = M[i, j] > 0.55 * vmax
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if hot else "#222222")
        ax.set_xticks(range(len(GAMMAS)), ["1", "2", "3", "C"])
        ax.get_xticklabels()[-1].set_color("#777777")
        ax.set_yticks(range(len(DELTAS)), [f"{d:.1f}" for d in DELTAS])
        ax.set_xlabel(r"budget $\Gamma$")
        ax.set_title(f"{n} targets", fontsize=9, pad=6)
        ax.tick_params(length=0)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if k == 0:
            ax.set_ylabel(r"degradation $\delta$")
        row, col = TRAIN_CELL
        ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, ec="#000000",
                               lw=1.4, zorder=5))
        style.panel_tag(ax, f"({chr(ord('a') + k)})", dx=-0.04, dy=1.14)

    fig.subplots_adjust(left=0.075, right=0.885, bottom=0.24, top=0.86, wspace=0.22)
    cax = fig.add_axes([0.91, 0.24, 0.011, 0.62])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("relative spread within the band (%)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_linewidth(0.4)

    require_matplotlib_panel_alignment(
        fig, json_out=str(style.QA_DIR / "figS1_grid.alignment.json"),
        overlay_svg=str(style.QA_DIR / "figS1_grid.alignment.svg"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=False, exclude_axes=(cax,))
    style.save(fig, "figS1_grid")

    for n in io.SCALES:
        M = mats[n]
        print(f"[figS1] {n} tgt: " + " | ".join(
            f"d{int(round(d * 100)):02d} " + " ".join(f"{M[i, j]:5.1f}" for j in range(4))
            for i, d in enumerate(DELTAS)))
    print(f"[figS1] global vmax = {vmax:.2f}%")
    for n in io.SCALES:
        M = mats[n]
        mono_d = all(np.all(np.diff(M[:, j]) > 0) for j in range(M.shape[1]))
        mono_g = all(np.all(np.diff(M[i, :]) > 0) for i in range(M.shape[0]))
        print(f"[figS1] {n} tgt monotone: delta={mono_d} gamma={mono_g}; "
              f"d30_g2={M[TRAIN_CELL]:.2f}%")


if __name__ == "__main__":
    main()
