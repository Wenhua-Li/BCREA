# -*- coding: utf-8 -*-
"""Fig.8 识别误差下的换案质量：误识别率 → 上场方案期望值（认证 RC 是任意识别的保证下限）。

图级主张：菜单换案对模式误识别退化平缓——10% 误识别保留 94–99% 的
（完美识别 − 认证 RC 下限）增益；任意识别规则都不会低于认证 RC。
数据契约（v3 分析层，纯后处理，无新搜索）：
- 20261013_v3_analysis/v3_review_ident.csv（analyze_v3_review.py --stage c）：
  bcrea_band m=5 菜单，6 演练实例 × 5 种子，两误差模型（uniform 完全错认 /
  shared 认出一半分量），p∈{0,…,1}，每单元 400 次蒙特卡洛；
  rc_floor 行 = 菜单认证 RC（p 列无意义，恒 0）。
- n：30 单元/规模；曲线=单元中位，带=IQR。
运行：py -3.11 figures/fig8_ident.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style
style.apply()
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

io = style.load_io()

P_GRID = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
MODELS = (("uniform", "completely wrong mode"), ("shared", "one component identified"))
COLORS = {24: style.METHOD["nominal_kcea"]["color"], 48: style.OI["sky"], 100: style.METHOD["bcrea_band"]["color"]}


def main() -> None:
    d = pd.read_csv(io.ANALYSIS_DIR / "v3_review_ident.csv")
    fig, axes = plt.subplots(1, 2, figsize=(style.DOUBLE_COL, 62 * style.MM))
    for ax, (model, label) in zip(axes, MODELS):
        for n in io.SCALES:
            g = d[(d.n_target == n) & (d.model == model)].sort_values("p")
            floor = d[(d.n_target == n) & (d.model == "rc_floor")]["mean_fielded_median"].iloc[0]
            ax.plot(g.p, g.mean_fielded_median, color=COLORS[n], label=f"{n} targets")
            ax.fill_between(g.p, g.mean_fielded_q25, g.mean_fielded_q75,
                            color=COLORS[n], alpha=0.18, lw=0)
            ax.axhline(floor, color=COLORS[n], ls=":", lw=0.8)
        ax.set_xlabel("mode misidentification rate $p$")
        ax.set_title(label, fontsize=9, pad=4)
        ax.set_xticks(P_GRID)
        ax.set_xticklabels(["0", ".05", ".1", ".2", ".4", ".6", ".8", "1"])
        ax.set_xlim(-0.02, 1.02)
    axes[0].set_ylabel("expected destroyed targets of fielded plan")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3,
               handlelength=1.6, columnspacing=1.2, bbox_to_anchor=(0.5, -0.005))
    for i, ax in enumerate(axes):
        style.panel_tag(ax, f"({chr(ord('a') + i)})", dx=-0.16, dy=1.04)
    fig.tight_layout(w_pad=1.4, rect=(0, 0.07, 1, 1))
    require_matplotlib_panel_alignment(fig)
    style.save(fig, "fig8_ident")
    for n in io.SCALES:
        g = d[(d.n_target == n) & (d.model == "uniform")]
        print(f"{n} tgt uniform: " + ", ".join(f"p={p}: {v:.2f}" for p, v in zip(g.p, g.mean_fielded_median)))


if __name__ == "__main__":
    main()
