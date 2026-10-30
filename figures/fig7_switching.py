# -*- coding: utf-8 -*-
"""Fig.7 换案演练与实测时延：菜单换案≈真实重解的质量，代价是微秒级查表。

图级主张：预认证小菜单的逐模式质量贴近真实重优化（100 目标 69% 角点完全打平、中位差 0），
而查表换案比单次角点重解快 7–8 个数量级；不换案的代价随规模放大。
数据契约（v3 分析层，全部实测/存档，无外推）：
- (a) 逐角点配对质量差：20261013_v3_analysis/v3_menu_switch.csv（6 实例 × 5 种子 × 66 角点），
  method=bcrea_band、m=5；三种策略相对真实重解的损失（kills，≥0，越小越好）：
  名义池不换案（nominal_stick_qual）/ BCREA 池不换案（stick_qual）/ BCREA 菜单换案（switch_qual）。
  注意：100 目标所有角点重解都到 72.0 封顶，所以"打平"= 菜单也到 72.0。
- (b) 分阶段实测墙钟：v3_timing.csv；实心 = 本机实测（查表/提取/MILP/读档），
  斜纹 = 战役存档实测（质量矩阵现算、单次角点重解）。**不含任何外推柱**。
  测量时机器有外部负载（见 v3_timing_env.txt），实际应更快。
- n：角点配对 660/规模；计时每段 18–660 个样本（见打印）。
统计：中位数+IQR；图注沿用脚本打印的计数。
运行：py -3.11 code/figure/fig7_switching.py
"""
import style
style.apply()
import matplotlib.pyplot as plt
import numpy as np

from audit_panel_alignment import require_matplotlib_panel_alignment

io = style.load_io()
C = style.OI

MENU_M = 5
STRATEGIES = (
    ("nominal_stick_qual", "nominal pool, no switch", style.METHOD["nominal_kcea"]["color"], 1.0),
    ("stick_qual", "BCREA pool, no switch", style.OI["sky"], 1.0),
    ("switch_qual", "BCREA menu, switch", style.METHOD["bcrea_band"]["color"], 1.0),
)
JITTER_ANGLE = 2.39996323          # 黄金角：确定性抖动，重跑逐点一致

TIMING_ROWS = (
    ("lookup_switch", "lookup + switch (66×m argmax)", False),
    ("menu_extract", "menu extraction (greedy RC)", False),
    ("milp_certify", "MILP certification (exact $m^{*}$)", False),
    ("quality_parse", "read quality matrix (JSON parse)", False),
    ("quality_build_archive", "build quality matrix (66 scenarios)", True),
    ("corner_reopt", "one corner re-solve (KCEA, 9000N)", True),
)


def load_switch() -> dict:
    s = io.load_analysis("v3_menu_switch")
    s = s[(s.method == "bcrea_band") & (s.m == MENU_M)]
    assert len(s) == 6 * 5 * 66, f"换案配对行数 {len(s)} != 1980"
    out = {}
    for n in io.SCALES:
        g = s[s.n_target == n]
        out[n] = {key: (g.reopt_best - g[key]).to_numpy(float) for key, *_ in STRATEGIES}
    return out


def load_timing() -> dict:
    t = io.load_analysis("v3_timing")
    out = {}
    for seg, _label, _arch in TIMING_ROWS:
        s = t[t.segment == seg]
        row = {}
        for n in io.SCALES:
            sn = s[s.n_target == n]
            if len(sn):
                row[n] = (float(sn.median_s.median()), float(sn.q25_s.median()),
                          float(sn.q75_s.median()), int(sn.n_samples.iloc[0]))
        if not row:                       # 只有汇总行（lookup/extract 的跨规模中位）
            row[100] = (float(s.median_s.median()), float(s.q25_s.median()),
                        float(s.q75_s.median()), int(s.n_samples.iloc[0]))
        out[seg] = row
    return out


def panel_loss(ax, d: dict, n_target: int, ylim: float):
    rng_x = np.arange(len(STRATEGIES))
    for i, (key, _label, color, _a) in enumerate(STRATEGIES):
        v = d[key]
        bp = ax.boxplot([v], positions=[i], widths=0.62, showfliers=False,
                        whis=(5, 95), patch_artist=True)
        for box in bp["boxes"]:
            box.set(facecolor="white", edgecolor=color, linewidth=0.9)
        for part in ("whiskers", "caps"):
            for line in bp[part]:
                line.set(color=color, linewidth=0.8)
        for line in bp["medians"]:
            line.set(color=color, linewidth=1.8)
        x = i + 0.20 * np.sin(np.arange(v.size) * JITTER_ANGLE)
        ax.plot(x, v, ls="", marker="o", ms=1.1, mfc=color, mec="none", alpha=0.22, zorder=1)
    ax.set_xticks(rng_x, ["nominal\nno switch", "BCREA\nno switch", "BCREA\nmenu + switch"])
    ax.set_xlim(-0.6, len(STRATEGIES) - 0.4)
    ax.set_ylim(-0.1, ylim)
    ax.set_title(f"{n_target} targets\n($n$={d['switch_qual'].size} corners)", fontsize=9, pad=4)


def panel_timing(ax, t: dict):
    # 对数时间轴的前提：所有计时中位数严格为正（墙钟正值，不会出现 0）
    meds = np.asarray([v[0] for row in t.values() for v in row.values()], float)
    assert not np.any(meds <= 0), "log time axis needs strictly positive medians"
    ys = np.arange(len(TIMING_ROWS))[::-1]
    xmin = 1e-6
    for y, (seg, label, archive) in zip(ys, TIMING_ROWS):
        med, q25, q75, _n = t[seg][100]
        ax.plot([q25, q75], [y, y], color="#333333", lw=2.0, solid_capstyle="butt", zorder=3)
        ax.plot(med, y, marker="s", ms=5.0, mfc="white" if archive else "#333333",
                mec="#333333", mew=0.9, zorder=4)
        ax.text(max(q75, med) * 1.35, y, _fmt_time(med), fontsize=8, va="center",
                ha="left")
        ax.text(xmin, y + 0.26, label + (" †" if archive else ""), fontsize=8,
                va="bottom", ha="left")
        if 24 in t[seg] and abs(t[seg][24][0] - med) > 1e-12:
            ax.plot(t[seg][24][0], y, marker="|", ms=6, color="#777777", zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(xmin, 3e4)
    ax.set_ylim(-1.05, len(TIMING_ROWS) - 0.15)
    ax.set_yticks([])
    ax.set_xticks([1e-6, 1e-4, 1e-2, 1e0, 1e2])
    ax.set_xticklabels(["1 µs", "100 µs", "10 ms", "1 s", "100 s"])
    ax.set_xlabel("wall-clock time (log scale)")
    # 横轴是数值轴，竖直网格线会穿过行内标签；这里只保留十进位刻度线，不加网格
    ax.grid(False)
    ax.minorticks_on()
    ax.tick_params(axis="x", which="minor", length=1.6, width=0.4, color="#888888")
    ax.text(xmin * 3.0, -0.62, "† campaign archive; others measured on this machine; "
            "grey tick = 24-target median", fontsize=8, color="#555555", va="center")


def _fmt_time(v: float) -> str:
    if v < 1e-3:
        return f"{v * 1e6:.1f} µs"
    if v < 1.0:
        return f"{v * 1e3:.1f} ms"
    return f"{v:.0f} s"


def main():
    style.require_batch()
    style.QA_DIR.mkdir(parents=True, exist_ok=True)
    sw = load_switch()
    tm = load_timing()

    ymax = max(float(np.quantile(v, 0.995)) for n in io.SCALES
               for v in sw[n].values())

    fig = plt.figure(figsize=(style.DOUBLE_COL, 108 * style.MM))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.88, 1.12],
                          left=0.06, right=0.985, bottom=0.145, top=0.93,
                          wspace=0.30, hspace=0.55)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for i, (ax, n) in enumerate(zip(axes, io.SCALES)):
        panel_loss(ax, sw[n], n, ymax)
        style.panel_tag(ax, f"({chr(ord('a') + i)})", dx=-0.02, dy=1.02)
        if i == 0:
            ax.set_ylabel("loss vs true re-solve (targets)")
    ax_t = fig.add_subplot(gs[1, :])
    panel_timing(ax_t, tm)
    style.panel_tag(ax_t, "(d)", dx=-0.02, dy=1.02)

    handles = [plt.Line2D([], [], marker="s", ls="", ms=5, mfc="white", mec=col,
                          mew=0.9, label=lab) for _k, lab, col, _a in STRATEGIES]
    handles.append(plt.Line2D([], [], marker="s", ls="", ms=5, mfc="white",
                              mec="#333333", mew=0.9, label="campaign runs"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.2, columnspacing=1.8)

    require_matplotlib_panel_alignment(
        fig, json_out=str(style.QA_DIR / "fig7_switching.alignment.json"),
        overlay_svg=str(style.QA_DIR / "fig7_switching.alignment.svg"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=False)
    style.save(fig, "fig7_switching")

    for n in io.SCALES:
        msg = [f"[fig7] N={n}: corners={sw[n]['switch_qual'].size}"]
        for key, label, _c, _a in STRATEGIES:
            v = sw[n][key]
            msg.append(f"{label}: med {np.median(v):.2f} IQR "
                       f"{np.quantile(v, 0.25):.2f}-{np.quantile(v, 0.75):.2f}")
        tie = float(np.mean(np.abs(sw[n]["switch_qual"]) <= 1e-9))
        msg.append(f"menu ties reopt {tie * 100:.1f}%")
        print(" | ".join(msg))
    print("[fig7] timing (median s @100 / IQR / n):")
    for seg, label, arch in TIMING_ROWS:
        med, q25, q75, n = tm[seg][100]
        extra = ""
        if 24 in tm[seg] and abs(tm[seg][24][0] - med) > 1e-12:
            extra = f", 24-target {tm[seg][24][0]:.4g}"
        print(f"   {label:38s} {med:.4g} [{q25:.4g}, {q75:.4g}] n={n}"
              f"{' (archive)' if arch else ''}{extra}")


if __name__ == "__main__":
    main()
