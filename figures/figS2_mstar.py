# -*- coding: utf-8 -*-
"""Fig. S2 认证覆盖数 m* 的分布（v3 批次，δ=30%，ε=1%，MILP/HiGHS 精确解）。

图级主张：m* 的分布随规模、方法与 Γ 结构性变化——24 目标几乎全部塌为 1；
48 目标 BCREA 两变体的中位升到 2；100 目标名义池中位 3（尾部到 7），
而鲁棒感知池把它压回 1–2；Γ=C 是单场景角点，m*=1 是代数必然。
数据契约：20261013_v3_analysis/v3_mstar_dist.csv（72 行 = 3 规模 × 6 方法 × 4 个 Γ，
每行给出该格 m*=1..12 的单元计数；单元 = 实例 × 种子，n = 160/70/70）。
画法：每格一行一个方法的 100% 叠条（颜色深浅 = m*），条右侧数字为该格中位 m*；
不做跨方法的显著性主张（显著性在正文 Table VI 与 SI Table S2）。
运行：py -3.11 code/figure/figS2_mstar.py
"""
import style
style.apply()
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from audit_panel_alignment import require_matplotlib_panel_alignment

io = style.load_io()

METHOD_ORDER = ["nominal", "pessimistic", "nsga2", "sampled_wc", "bcrea_band", "bcrea_strict"]
SHORT = {"nominal": "Nominal", "pessimistic": "Pessim.",
         "nsga2": "NSGA-II", "sampled_wc": "Sampled",
         "bcrea_band": "Band", "bcrea_strict": "Strict"}
GAMMAS = (1, 2, 3, 12)
MAXM = 7
# 深浅 = 菜单规模：m*=1 最浅（"不需要菜单"），m* 越大越深
SHADES = plt.get_cmap("Blues")(np.linspace(0.12, 0.92, MAXM))


def load_counts() -> dict[tuple[int, str, int], np.ndarray]:
    """(规模, 方法, Γ) → m*=1..7 的计数向量。"""
    d = io.load_analysis("v3_mstar_dist")
    cols = [f"n_mstar_{k}" for k in range(1, MAXM + 1)]
    out = {}
    for _, r in d.iterrows():
        key = (int(r.n_target), r.method, int(r.gamma))
        out[key] = np.asarray([float(r[c]) for c in cols], dtype=float)
    return out


def median_of(counts: np.ndarray) -> int:
    cum = np.cumsum(counts)
    return int(np.searchsorted(cum, 0.5 * cum[-1]) + 1)


def panel(ax, counts_by_method: dict[str, np.ndarray], gamma: int, n_units: int,
          show_labels: bool):
    for row, method in enumerate(METHOD_ORDER):
        y = len(METHOD_ORDER) - 1 - row
        c = counts_by_method[method]
        share = c / c.sum() * 100.0
        left = 0.0
        for k in range(MAXM):
            if share[k] <= 0:
                continue
            # 段间留 0.8 个百分点的白缝（无描边：描边会被文字碰撞审计判为
            # text-stroke，白缝同样可分辨且不产生 stroke path）
            gap = 0.8 if share[k] > 2.0 else 0.0
            ax.barh(y, max(share[k] - gap, 0.1), left=left + gap / 2, height=0.80,
                    color=SHADES[k], edgecolor="none")
            if share[k] >= 9.0:
                ax.text(left + share[k] / 2, y, f"{share[k]:.0f}", ha="center",
                        va="center", fontsize=8,
                        color="white" if k >= 3 else "#222222")
            left += share[k]
        ax.text(108.0, y, f"{median_of(c)}", ha="left", va="center", fontsize=8,
                color="#000000", fontweight="bold")
    ax.text(108.0, len(METHOD_ORDER) - 0.45, "med", ha="left", va="bottom",
            fontsize=8, color="#555555")
    ax.set_xlim(0, 116)
    ax.set_ylim(-0.6, len(METHOD_ORDER) - 0.4)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks(range(len(METHOD_ORDER)),
                  [SHORT[m] for m in METHOD_ORDER][::-1] if show_labels else [])
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"$\\Gamma$ = {gamma}  ($n$ = {n_units})", fontsize=9, pad=4)
    ax.grid(False)


def main():
    style.require_batch()
    style.QA_DIR.mkdir(parents=True, exist_ok=True)
    counts = load_counts()

    fig, axs = plt.subplots(3, 4, figsize=(175 * style.MM, 132 * style.MM))
    for i, n in enumerate(io.SCALES):
        for j, gm in enumerate(GAMMAS):
            ax = axs[i, j]
            per = {m: counts[(n, m, gm)] for m in METHOD_ORDER}
            n_units = int(sum(per["nominal"]))
            panel(ax, per, gm, n_units, show_labels=(j == 0))
            if i == len(io.SCALES) - 1:
                ax.set_xlabel("share of units (%)")
            if j == 0:
                ax.set_ylabel(f"{n} targets", fontsize=9)
            if i < len(io.SCALES) - 1:
                ax.tick_params(axis="x", labelbottom=False)
        style.panel_tag(axs[i, 0], f"({chr(ord('a') + i)})", dx=-0.46, dy=1.13)

    fig.subplots_adjust(left=0.082, right=0.985, bottom=0.155, top=0.93,
                        wspace=0.14, hspace=0.50)
    handles = [Patch(facecolor=SHADES[k], edgecolor="white", linewidth=0.4,
                     label=f"$m^{{*}}$ = {k + 1}") for k in range(MAXM)]
    fig.legend(handles=handles, loc="lower center", ncol=MAXM, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=1.2,
               columnspacing=1.4, handletextpad=0.5)

    require_matplotlib_panel_alignment(
        fig, json_out=str(style.QA_DIR / "figS2_mstar.alignment.json"),
        overlay_svg=str(style.QA_DIR / "figS2_mstar.alignment.svg"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=False)
    style.save(fig, "figS2_mstar")

    for n in io.SCALES:
        for gm in GAMMAS:
            meds = {m: median_of(counts[(n, m, gm)]) for m in METHOD_ORDER}
            n_units = int(sum(counts[(n, "nominal", gm)]))
            print(f"[figS2] {n:3d} tgt G={gm:<2d} n={n_units}: " +
                  " ".join(f"{SHORT[m]}={meds[m]}" for m in METHOD_ORDER))
    allc = np.array([counts[k] for k in counts])
    print(f"[figS2] overall P(m*=1)={allc[:, 0].sum() / allc.sum():.3f}, "
          f"max m*={int(np.nonzero(allc.sum(axis=0))[0].max()) + 1}")


if __name__ == "__main__":
    main()
