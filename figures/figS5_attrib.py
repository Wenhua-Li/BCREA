# -*- coding: utf-8 -*-
"""Fig. S5 最坏场景的部件归因（v3 批次，δ=30%，Γ=2，离线重算）。

图级主张：带内方案的最坏场景不是均匀落在 12 个能力分量上——24/48 目标上发射车杀伤边界
（K1–K6）吃掉 65–88% 的归因事件，且前两类发射车（K1/K2）单独占三到四成；
到 100 目标，BCREA 两变体的归因摊平到雷达与杀伤边界两侧（K 侧 40–46%），
即没有任何一个分量族单独扛住风险。归因是"哪个分量是这条方案最坏场景的元凶"的描述统计，
不是因果分解，也不跨方法做显著性主张。
数据契约：20261013_v3/scans/<method>_<instance>_s<seed>.json 的 cells.d30_g2
（objsS 66×池、masks 66×12、argmin 每方案最坏场景）+ nominal.eq_idx（该池名义等价带）。
口径（同 analyze_structure.py 的 worst_component_hits）：对等价带内每个成员取其最坏场景
（objsS 上 argmin），把该场景掩码里为 1 的分量计一次；份额 = 分量命中 / 全部命中。
Γ=2 时每次归因恰好 2 个分量，故均匀无集中的基准是 1/6 = 16.7%（图中竖虚线）。
运行：py -3.11 code/figure/figS5_attrib.py（约 30 s：900 份扫描 JSON）
"""
import numpy as np

import style
style.apply()
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from audit_panel_alignment import require_matplotlib_panel_alignment

io = style.load_io()

METHODS = ["nominal_kcea", "bcrea_band", "bcrea_strict"]
LABELS = [f"R{a + 1}" for a in range(6)] + [f"K{b + 1}" for b in range(6)]
CELL = "d30_g2"


def attribution() -> tuple[dict[int, dict[str, np.ndarray]], int]:
    """逐规模累计 12 个分量的命中次数（带内成员 × 最坏场景掩码）。"""
    hits: dict[int, dict[str, np.ndarray]] = {
        n: {m: np.zeros(12) for m in METHODS} for n in io.SCALES}
    n_files = 0
    for method in METHODS:
        for n in io.SCALES:
            for inst in io.instances(n):
                for seed in io.SEEDS:
                    scan = io.load_scan(method, inst, seed)
                    cell = scan["cells"][CELL]
                    masks = np.asarray(cell["masks"], dtype=int)
                    argmin = np.asarray(cell["argmin"], dtype=int)
                    eq = np.asarray(scan["nominal"]["eq_idx"], dtype=int)
                    worst = masks[argmin[eq]]                  # (|E|, 12)
                    hits[n][method] += worst.sum(axis=0)
                    n_files += 1
    return hits, n_files


def panel(ax, per_method: dict[str, np.ndarray], n_target: int):
    y = np.arange(len(LABELS))[::-1]
    h = 0.26
    offs = {"nominal_kcea": +h, "bcrea_band": 0.0, "bcrea_strict": -h}
    total_events = {}
    for method in METHODS:
        c = per_method[method]
        share = c / c.sum() * 100.0
        total_events[method] = c.sum() / 2
        m = style.METHOD[method]
        # strict 用同色浅填充（不用 hatch：hatch 在 PDF 里是描边图案，会被文字碰撞审计
        # 判成贯穿整页的 stroke path）；灰度下浅填充与 band 的实色仍可区分
        ax.barh(y + offs[method], share, height=h * 0.92, color=m["color"],
                edgecolor="none", alpha=0.42 if method == "bcrea_strict" else 1.0)
    ax.plot([100.0 / 6.0] * 2, [-0.5, 11.4], color="#777777", lw=style.LW_REF,
            ls=(0, (2.4, 1.6)), zorder=1)
    ax.set_yticks(y, LABELS)
    ax.set_xlim(0, 40)
    ax.set_ylim(-0.6, 12.3)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xlabel("share of worst-case attributions (%)")
    ax.set_title(f"{n_target} targets", fontsize=9, pad=5)
    ax.grid(False, axis="y")
    ax.tick_params(axis="y", length=0)
    # R/K 分区：分隔线 + 右缘竖排说明（R 组在上：y=11..6；K 组在下：y=5..0）。
    # K 组说明拆两行竖排：单行 "engagement boundaries" 太长会伸进 R 组标签区（视觉验收抓过叠印）
    ax.axhline(5.5, color="#999999", lw=0.6, ls=":", zorder=1)
    ax.text(38.5, 2.75, "engagement\nboundaries", fontsize=8, color="#555555", rotation=90,
            rotation_mode="anchor", ha="center", va="center", linespacing=1.1)
    ax.text(38.5, 8.5, "radar ranges", fontsize=8, color="#555555", rotation=90,
            rotation_mode="anchor", ha="center", va="center")
    ax.text(39.5, 11.35, "events " + "/".join(
        f"{total_events[m] / 1000:.1f}k" for m in METHODS), fontsize=8, color="#333333",
        ha="right", va="bottom")
    return total_events


def main():
    style.require_batch()
    style.QA_DIR.mkdir(parents=True, exist_ok=True)
    hits, n_files = attribution()
    print(f"[figS5] scanned {n_files} scan files (30 instances x 10 seeds x 3 methods)")

    fig, axs = plt.subplots(1, 3, figsize=(191 * style.MM, 66 * style.MM))
    for k, (ax, n) in enumerate(zip(axs, io.SCALES)):
        ev = panel(ax, hits[n], n)
        style.panel_tag(ax, f"({chr(ord('a') + k)})", dx=-0.02, dy=1.10)
        if k > 0:
            ax.set_yticklabels([])
        print(f"[figS5] {n:3d} tgt events " + ", ".join(f"{m}={ev[m]:.0f}" for m in METHODS))
        for m in METHODS:
            sh = hits[n][m] / hits[n][m].sum() * 100
            print("        " + f"{m:14s} " + " ".join(f"{v:5.1f}" for v in sh))
    handles = [Patch(facecolor=style.METHOD[m]["color"],
                     alpha=0.42 if m == "bcrea_strict" else 1.0,
                     label=style.METHOD[m]["label"]) for m in METHODS]
    handles.append(plt.Line2D([], [], color="#777777", lw=style.LW_REF, ls=(0, (2.4, 1.6)),
                              label="uniform baseline (1/6)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.8, columnspacing=2.2)

    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.26, top=0.89, wspace=0.14)
    require_matplotlib_panel_alignment(
        fig, json_out=str(style.QA_DIR / "figS5_attrib.alignment.json"),
        overlay_svg=str(style.QA_DIR / "figS5_attrib.alignment.svg"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=False)
    style.save(fig, "figS5_attrib")

    for n in io.SCALES:
        row = []
        for method in METHODS:
            sh = hits[n][method] / hits[n][method].sum() * 100
            row.append(f"{style.METHOD[method]['label']}: K-share {sh[6:].sum():.1f}% "
                       f"max {LABELS[int(sh.argmax())]}={sh.max():.1f}%")
        print(f"[figS5] {n:3d} tgt | " + " | ".join(row))


if __name__ == "__main__":
    main()
