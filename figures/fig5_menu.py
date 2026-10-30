# -*- coding: utf-8 -*-
"""Fig.5 认证菜单：少量方案即可保住池的鲁棒覆盖，MILP 给出最小覆盖数。

图级主张：菜单大小 m 增大带来的覆盖增益很快饱和（m=1–2 就吃掉整池 RC 的绝大部分），
但"要不要菜单"随规模与池而异——精确 m* 把它变成可查的数字。
数据契约（v3 正式批次 + 分析层）：
- (a) RC(m)：scans 的 cells.d30_g2.objsS 现算（io.greedy_rc_select 取 10 个成员，
  逐前缀算 min_s max_{x∈X_m} f_s(x)）；三方法 nominal/band/strict × 三规模；
  纵轴画"相对整池 RC 的缺口" RC(m) − rc_full_pool（≤0），y=0 即整池 RC；
  最优单解 WC 以点线给出（与整池 RC 的差，|gap|≤0.05 时省略）。
  脚本内与 20261013_v3_analysis/v3_rc.csv 的 rc5 逐 run 核对（必须逐位一致）。
- (b) 精确覆盖数 m*：20261013_v3_analysis/v3_mstar.csv 的 mstar_exact，
  三方法 × 三规模 × Γ∈{1,2,3,C}；格内 = 中位 m* 与 P(m*=1)；Γ=C 列是单场景代数边界
  （灰斜纹），不是"实测更难"的证据。
- n：每格 10 种子 × 实例数（24 目标 16 实例、48/100 目标各 7 实例）；RC(m) 不剔除 run，
  等价带只有 k 个成员的 run 只贡献 m≤k 的点，所以 m=1 用 900 个 run、m=10 用 836 个
  （带内成员 <5 的 23 个、<10 的 64 个）。
统计：中位数+IQR；图注沿用脚本打印的计数。
运行：py -3.11 code/figure/fig5_menu.py
"""
import style
style.apply()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from audit_panel_alignment import require_matplotlib_panel_alignment

io = style.load_io()
C = style.OI

MENU_METHODS = ("nominal_kcea", "bcrea_band", "bcrea_strict")
M_MAX = 10
GAMMAS = (1, 2, 3, 12)
BAR_M = (1, 3, 5, 10)          # 画 IQR 的 m 值


def load_rc_curves() -> tuple[pd.DataFrame, int, int]:
    """逐 run 的 RC(m), m=1..M_MAX 与三规模；顺带核对 v3_rc.csv 的 rc5。

    不剔除 run：等价带只有 k 个成员时该 run 只贡献 m≤k 的点（曲线按 m 取可得样本）；
    返回 (表, 带内成员 <5 的 run 数, 带内成员 <10 的 run 数)。"""
    rows = []
    n_lt5 = n_lt10 = 0
    for method in MENU_METHODS:
        for n in io.SCALES:
            for inst in io.instances(n):
                for seed in io.SEEDS:
                    arr = io.scan_arrays(io.load_scan(method, inst, seed), "d30_g2")
                    S, eq = arr["objsS"], arr["eq_idx"]
                    n_lt5 += int(len(eq) < 5)
                    n_lt10 += int(len(eq) < M_MAX)
                    chosen = io.greedy_rc_select(S[:, eq], M_MAX)
                    cur = np.full(S.shape[0], -np.inf)
                    pool = float(arr["cell"]["rc_full_pool"])
                    single = float(arr["cell"]["best_single_wc"])
                    for k, idx in enumerate(chosen, 1):
                        cur = np.maximum(cur, S[:, eq[idx]])
                        rows.append({"method": method, "n_target": n, "instance": inst,
                                     "seed": seed, "m": k, "rc": float(cur.min()),
                                     "rc_pool": pool, "best_single": single})
    df = pd.DataFrame(rows)

    ref = io.load_analysis("v3_rc")
    ref["method"] = ref.method.map(io.PREFIX_METHOD)
    ref = ref[ref.method.isin(MENU_METHODS)]
    got = df[df.m == 5].set_index(["method", "instance", "seed"]).rc
    want = ref.set_index(["method", "instance", "seed"]).rc5
    joined = got.to_frame("got").join(want.rename("want"), how="inner")
    assert len(joined) == len(ref) - n_lt5, \
        f"rc5 核对行数不符: {len(joined)} vs {len(ref)}-{n_lt5}"
    assert (joined.got - joined.want).abs().max() < 1e-9, "RC(5) 与 v3_rc.csv 不一致"
    return df, n_lt5, n_lt10


def load_mstar() -> pd.DataFrame:
    ms = io.load_analysis("v3_mstar")
    ms["method"] = ms.method.map(io.PREFIX_METHOD)
    ms = ms[ms.method.isin(MENU_METHODS)]
    assert ms.milp_ok.all(), "存在 MILP 未解出的格，不能画精确 m*"
    return ms


def panel_rc(ax, sub: pd.DataFrame, n_target: int):
    ax.axhline(0.0, color="#555555", lw=style.LW_REF, ls="--", zorder=1)
    for method in MENU_METHODS:
        s = sub[sub.method == method]
        spec = style.METHOD[method]
        sf = s.assign(sf=s.rc - s.rc_pool)          # 逐 run 配对：相对整池 RC 的缺口
        med = sf.groupby("m").sf.median()
        q25 = sf.groupby("m").sf.quantile(0.25)
        q75 = sf.groupby("m").sf.quantile(0.75)
        ax.plot(med.index, med.values, color=spec["color"], ls=spec["ls"], lw=style.LW_DATA,
                zorder=3)
        sel = [m for m in BAR_M if m in med.index]
        ax.errorbar(sel, med.loc[sel].values,
                    yerr=[med.loc[sel].values - q25.loc[sel].values,
                          q75.loc[sel].values - med.loc[sel].values],
                    fmt="none", ecolor=spec["color"], elinewidth=0.8, capsize=1.6,
                    capthick=0.8, zorder=2)
        gap = float((s.rc_pool - s.best_single).median())
        if gap > 0.05:                       # 最优单解明显低于整池 RC 时画点线
            ax.axhline(-gap, color=spec["color"], lw=style.LW_REF, ls=":", alpha=0.9,
                       zorder=1)
    ax.set_xlim(0.7, M_MAX + 0.3)
    ax.set_xticks([1, 2, 4, 6, 8, 10])
    ax.set_xlabel("menu size $m$")
    ax.set_ylim(-1.65, 0.28)
    ax.set_title(f"{n_target} targets\n($n$={sub.groupby(['method', 'instance', 'seed']).ngroups} runs)",
                 fontsize=9, pad=4)


def panel_mstar(ax, sub: pd.DataFrame, method: str):
    med = np.zeros((len(io.SCALES), len(GAMMAS)))
    p1 = np.zeros_like(med)
    for i, n in enumerate(io.SCALES):
        for j, g in enumerate(GAMMAS):
            s = sub[(sub.method == method) & (sub.n_target == n) & (sub.gamma == g)]
            med[i, j] = float(np.median(s.mstar_exact))
            p1[i, j] = float(np.mean(s.mstar_exact == 1))
    im = ax.imshow(med, cmap="Blues", vmin=1.0, vmax=3.0, aspect="auto")
    for i in range(med.shape[0]):
        for j in range(med.shape[1]):
            dark = med[i, j] >= 2.5
            ax.text(j, i - 0.13, f"{med[i, j]:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if dark else "black", fontweight="bold")
            ax.text(j, i + 0.22, f"{p1[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if dark else "#333333")
    # Γ=C 列：单场景代数边界（灰斜纹，非"更难"证据）
    ax.add_patch(Rectangle((2.5, -0.5), 1.0, len(io.SCALES), facecolor="none",
                           edgecolor="#999999", hatch="////", lw=0.0, zorder=2))
    ax.set_xticks(range(len(GAMMAS)), ["1", "2", "3", "C"])
    ax.get_xticklabels()[-1].set_color("#777777")
    ax.set_yticks(range(len(io.SCALES)), [str(n) for n in io.SCALES])
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(style.METHOD[method]["label"], fontsize=9, pad=4)
    return im


def main():
    style.require_batch()
    style.QA_DIR.mkdir(parents=True, exist_ok=True)
    rc, n_lt5, n_lt10 = load_rc_curves()
    ms = load_mstar()

    fig = plt.figure(figsize=(style.DOUBLE_COL, 96 * style.MM))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.90, 0.95],
                          left=0.065, right=0.985, bottom=0.17, top=0.865,
                          wspace=0.28, hspace=0.52)
    axes_a = [fig.add_subplot(gs[0, i]) for i in range(3)]
    axes_b = [fig.add_subplot(gs[1, i]) for i in range(3)]
    for i, (ax, n) in enumerate(zip(axes_a, io.SCALES)):
        panel_rc(ax, rc[rc.n_target == n], n)
        style.panel_tag(ax, f"({chr(ord('a') + i)})", dx=-0.02, dy=1.02)
        if i == 0:
            ax.set_ylabel("ΔRC (targets)")
    handles = [plt.Line2D([], [], color=style.METHOD[m]["color"], ls=style.METHOD[m]["ls"],
                          lw=style.LW_DATA, label=style.METHOD[m]["label"])
               for m in MENU_METHODS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3,
               fontsize=8, frameon=False, handlelength=1.8, handletextpad=0.5,
               columnspacing=2.2)
    ims = [panel_mstar(ax, ms, m) for ax, m in zip(axes_b, MENU_METHODS)]
    for i, ax in enumerate(axes_b):
        style.panel_tag(ax, f"({chr(ord('d') + i)})", dx=-0.02, dy=1.06)
        if i == 0:
            ax.set_ylabel("targets $N$")
    for ax in axes_b:
        ax.set_xlabel("budget $\\Gamma$")

    cax = fig.add_axes([0.40, 0.055, 0.24, 0.018])
    cb = fig.colorbar(ims[0], cax=cax, orientation="horizontal")
    cb.set_label("exact covering number $m^{*}$ (MILP), median", fontsize=8)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_linewidth(0.4)

    require_matplotlib_panel_alignment(
        fig, json_out=str(style.QA_DIR / "fig5_menu.alignment.json"),
        overlay_svg=str(style.QA_DIR / "fig5_menu.alignment.svg"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=False, exclude_axes=(cax,))
    style.save(fig, "fig5_menu")

    for n in io.SCALES:
        sub = rc[rc.n_target == n]
        msg = [f"[fig5] N={n}:"]
        for m in MENU_METHODS:
            s = sub[sub.method == m]
            pool = float(s.rc_pool.median())
            rc1 = float(s[s.m == 1].rc.median())
            rc10 = float(s[s.m == 10].rc.median())
            single = float(s.best_single.median())
            msg.append(f"{m}: pool RC {pool:.2f}, RC1 {rc1:.2f}, RC10 {rc10:.2f}, "
                       f"best single {single:.2f}, n={s.groupby(['instance', 'seed']).ngroups}")
        print(" ".join(msg))
    for m in (1, 5, M_MAX):
        sub_m = rc[rc.m == m]
        print(f"[fig5] RC(m) m={m}: runs="
              + ", ".join(f"{k}={sub_m[sub_m.n_target == n].groupby(['method', 'instance', 'seed']).ngroups}"
                          for k, n in [("24", 24), ("48", 48), ("100", 100)]))
    print(f"[fig5] 等价带 <5 的 run: {n_lt5}；<10 的 run: {n_lt10}（不剔除，曲线按 m 取可得样本）")
    print("[fig5] m* median/P(m*=1):")
    for method in MENU_METHODS:
        s = ms[ms.method == method]
        for n in io.SCALES:
            row = []
            for g in GAMMAS:
                t = s[(s.n_target == n) & (s.gamma == g)]
                row.append(f"{np.median(t.mstar_exact):.0f}/{np.mean(t.mstar_exact == 1):.2f}")
            print(f"   {method:14s} N={n:3d}: " + "  ".join(row))
    print(f"[fig5] overall P(m*=1)={np.mean(ms.mstar_exact == 1):.4f}, "
          f"max m*={int(ms.mstar_exact.max())}, n_runs={len(ms)}")


if __name__ == "__main__":
    main()
