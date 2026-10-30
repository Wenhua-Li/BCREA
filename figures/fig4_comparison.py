# -*- coding: utf-8 -*-
"""Fig.4 六方法同预算主比较：名义损失 × 鲁棒覆盖（v3 正式批次）。

图级主张：同一真实解码预算 D=9000N 下，六方法占据名义-鲁棒权衡面的不同位置；
诚实呈现——sampled worst-case 在主工作点的 RC5 上并不弱，BCREA 的卖点是带认证保证。
数据契约（v3 pools 30 实例×6 方法×10 种子；主工作点 δ=0.3、Γ=2）：
- 名义损失：20261013_v3_analysis/v3_nominal.csv 的 best_nominal，对"同实例全部方法全部
  种子的 best-known 名义值"取相对损失（%）。best-known 不是已证最优，图注不得写 optimal；
- RC5：v3_rc.csv 的 rc5 = 在名义等价带成员里贪心选 5 个方案，在 66 个 Γ=2 扰动模式上的
  鲁棒覆盖 min_s max_{x∈X5} f_s(x)（analyze_v3.py 的 M_RC=5 口径）。
  **必须配 scans 的公共网格**：pools 里的 deg_wc 是各方法自家训练场景上的值，跨方法不可比，
  所以纵轴与横轴数值都取自扫描分析层，而不是池内自评。
- 布局：1×3 三规模列（24/48/100），每列逐 run 小点 + 实例中位点 + 方法中位（IQR 杠）；
  三列共用横轴（同一 % 标尺），100 目标列标 72.0 封顶参考线；纵轴按规模各用各的量程。
- 预算核对：所有 run 的实际解码量 ≤ 9000·N（不超支；sampled_wc 因代数块不整除预算，
  个别 run 比 cap 低约 0.2%），本脚本断言。
统计：中位数+IQR（全篇同一写法）；图不塞 p 值，配对检验只在表格。
运行：py -3.11 code/figure/fig4_comparison.py
"""
import style
style.apply()
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from audit_panel_alignment import require_matplotlib_panel_alignment

io = style.load_io()
C = style.OI

CAP_100 = 72.0          # 100 目标列的封顶值（所有 run 的名义最优都不超过它）
X_MAX = 16.5            # 三列共用横轴上限（最大损失 15.7%）


def load_runs():
    """逐 run 表：instance/method/seed/scale/loss/rc5，附带预算核对。"""
    nom = io.load_analysis("v3_nominal")
    rc = io.load_analysis("v3_rc")
    df = nom.merge(rc[["instance", "method", "seed", "rc5"]],
                   on=["instance", "method", "seed"], how="inner", validate="one_to_one")
    assert len(df) == 1800, f"逐 run 行数 {len(df)} != 1800"
    # 分析层 CSV 用短前缀（nominal/pessimistic/…），统一换回池 config 的长标签
    df["method"] = df.method.map(io.PREFIX_METHOD)
    assert df.method.notna().all(), "分析层出现未知方法前缀"
    assert set(df.method) == set(io.METHODS), sorted(set(df.method))
    # 预算口径：所有 run 的 cap 都是 9000·N，实际解码量不超支（sampled_wc 因代数块
    # 不整除预算，个别 run 略低于 cap，约 0.2%，不是超支）
    cap = 9000 * df.n_target
    assert (df.actual_decode_evals <= cap).all(), "实际解码量超过 9000·N 预算"
    short = float((cap - df.actual_decode_evals).max() / cap.max())
    assert short < 0.01, f"预算利用率缺口过大: {short:.3%}"

    ref = df.groupby("instance").best_nominal.max()      # best-known，不是已证最优
    df["ref"] = df.instance.map(ref)
    df["loss"] = 100.0 * (df.ref - df.best_nominal) / df.ref
    return df


def mstyle(method: str) -> dict:
    """band 实心、strict 空心菱形：同蓝色族的第二重区分（STYLE.md 第三节）。"""
    spec = style.METHOD[method]
    if method == "bcrea_strict":
        return {"marker": spec["marker"], "mfc": "white", "mec": spec["color"]}
    return {"marker": spec["marker"], "mfc": spec["color"], "mec": spec["color"]}


def scatter_panel(ax, sub, n_target: int):
    for method in io.METHODS:                            # 1) 逐 run 小点
        s = sub[sub.method == method]
        ax.plot(s.loss, s.rc5, ls="", marker="o", ms=1.3, mec="none",
                mfc=style.METHOD[method]["color"], alpha=0.22, zorder=1)
    for method in io.METHODS:                            # 2) 实例中位点
        s = sub[sub.method == method]
        g = s.groupby("instance")[["loss", "rc5"]].median()
        st = mstyle(method)
        ax.plot(g.loss, g.rc5, ls="", ms=3.2, mew=0.7, alpha=0.9, zorder=3,
                marker=st["marker"], mfc=st["mfc"], mec=st["mec"])
    for method in io.METHODS:                            # 3) 方法中位 + IQR
        s = sub[sub.method == method]
        x, y = s.loss.to_numpy(float), s.rc5.to_numpy(float)
        xm, ym = float(np.median(x)), float(np.median(y))
        xq = np.quantile(x, [0.25, 0.75])
        yq = np.quantile(y, [0.25, 0.75])
        st = mstyle(method)
        ax.errorbar([xm], [ym], xerr=[[xm - xq[0]], [xq[1] - xm]],
                    yerr=[[ym - yq[0]], [yq[1] - ym]], fmt=st["marker"], ms=6.5,
                    mfc=st["mfc"], mec=st["mec"], mew=1.0, ecolor=style.METHOD[method]["color"],
                    elinewidth=0.9, capsize=2.0, capthick=0.9, zorder=5)
    if n_target == 100:
        ax.axhline(CAP_100, color=C["grey"], lw=style.LW_REF, ls="--", zorder=2)
        ax.text(X_MAX - 0.4, CAP_100 - 0.12, f"ceiling {CAP_100:.1f}", fontsize=8,
                ha="right", va="top", color="#555555")
    ax.set_xlim(-0.4, X_MAX)
    ax.set_xticks([0, 4, 8, 12, 16])
    ax.set_title(f"{n_target} targets", fontsize=9, pad=4)


def main():
    style.require_batch()
    style.QA_DIR.mkdir(parents=True, exist_ok=True)
    df = load_runs()

    fig = plt.figure(figsize=(style.DOUBLE_COL, 72 * style.MM))
    gs = fig.add_gridspec(1, 3, left=0.068, right=0.99, bottom=0.31, top=0.90, wspace=0.26)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for i, (ax, n) in enumerate(zip(axes, io.SCALES)):
        scatter_panel(ax, df[df.n_target == n], n)
        ys = df.loc[df.n_target == n, "rc5"]
        pad = 0.05 * float(ys.max() - ys.min())
        ax.set_ylim(float(ys.min()) - pad, float(ys.max()) + pad)
        style.panel_tag(ax, f"({chr(ord('a') + i)})", dx=-0.02, dy=1.02)
    axes[0].set_ylabel("RC$_5$ (targets)")

    n_runs = "/".join(str(int((df.n_target == n).sum())) for n in io.SCALES)
    fig.text(0.53, 0.185,
             f"nominal loss vs best-known reference (%)   ·   $n$ = {n_runs} runs",
             ha="center", va="center", fontsize=9)
    order = [0, 3, 1, 4, 2, 5]      # ncol=3 按列填充：排成两行三列后仍是方法顺序
    handles = [Line2D([], [], ls="", ms=6.0, label=style.METHOD[io.METHODS[k]]["label"],
                      marker=mstyle(io.METHODS[k])["marker"],
                      mfc=mstyle(io.METHODS[k])["mfc"],
                      mec=mstyle(io.METHODS[k])["mec"], mew=0.9)
               for k in order]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.53, 0.01), frameon=False, handletextpad=0.5,
               columnspacing=1.4, labelspacing=0.45)

    require_matplotlib_panel_alignment(
        fig, json_out=str(style.QA_DIR / "fig4_comparison.alignment.json"),
        overlay_svg=str(style.QA_DIR / "fig4_comparison.alignment.svg"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=False)
    style.save(fig, "fig4_comparison")

    for n in io.SCALES:
        sub = df[df.n_target == n]
        print(f"[fig4] N={n}: runs={len(sub)} loss% max={sub.loss.max():.1f} "
              f"median={sub.loss.median():.2f}")
        for m in io.METHODS:
            s = sub[sub.method == m]
            print(f"       {m:22s} loss={s.loss.median():5.2f}%  rc5={s.rc5.median():6.2f}")


if __name__ == "__main__":
    main()
