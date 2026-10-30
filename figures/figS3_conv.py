# -*- coding: utf-8 -*-
"""Fig. S3 搜索过程：名义目标与各自鲁棒目标随真实解码预算的推进（v3 批次）。

图级主张：同一预算下，BCREA 的名义目标一路跟住名义池（面板 a，同一语义、可直接读），
而它自己的鲁棒目标（带内损失）在预算早期就被压到接近终值（面板 b）。
数据契约：20261013_v3_analysis/v3_traces.csv（50 个预算分位点，按 方法×序列×规模
给出 n_runs、中位与四分位；单元 = 实例 × 种子，n = 160/70/70）。
口径（必须写进图注，禁止跨方法比绝对值）：
- 面板 (a)：nominal_obj 对三种池同义（名义参数下的期望击杀），可直接比较；
  不含 NSGA-II（无 trace）。
- 面板 (b)：每条曲线是该方法**自己的**目标——band/strict = 带内损失（训练口径 K=8 场景，
  非 66 角点评估）、pessimistic = 悲观问题目标值、sampled_wc = 16 场景最坏适应度。
  语义与量纲都不同，所以只画"已闭合的自身改进缺口"（= 1 − |v(t) − v_end| / |v(0) − v_end|），
  不跨方法比较绝对值；v(0) 是第 0 个分位点、v_end 是第 50 个分位点的中位。
运行：py -3.11 code/figure/figS3_conv.py
"""
import style
style.apply()
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from audit_panel_alignment import require_matplotlib_panel_alignment

io = style.load_io()

SERIES_A = ["nominal_kcea", "bcrea_band", "bcrea_strict"]
SERIES_B = ["bcrea_band", "bcrea_strict", "pessimistic_resolve", "sampled_worst_case"]
SERIES_B_LABEL = {
    "bcrea_band": "BCREA (band): in-band loss (training)",
    "bcrea_strict": "BCREA (strict): in-band loss (training)",
    "pessimistic_resolve": "Pessimistic: pessimistic objective",
    "sampled_worst_case": "Sampled WC: 16-scenario worst fitness",
}
TRACE_KEY = {"nominal_kcea": ("nominal", "nominal_obj"),
             "bcrea_band": ("bcrea_band", "nominal_obj"),
             "bcrea_strict": ("bcrea_strict", "nominal_obj")}
TRACE_KEY_B = {"bcrea_band": ("bcrea_band", "in_band_deg"),
               "bcrea_strict": ("bcrea_strict", "in_band_deg"),
               "pessimistic_resolve": ("pessimistic", "pessimistic_obj"),
               "sampled_worst_case": ("sampled_wc", "sampled_wc_fit")}


def get(t, n: int, key: tuple[str, str]):
    s = t[(t.n_target == n) & (t.method == key[0]) & (t.series == key[1])]
    return s.sort_values("budget_frac")


def main():
    style.require_batch()
    style.QA_DIR.mkdir(parents=True, exist_ok=True)
    t = io.load_analysis("v3_traces")

    fig = plt.figure(figsize=(182 * style.MM, 108 * style.MM))
    gs = fig.add_gridspec(2, 3, left=0.075, right=0.985, top=0.92, bottom=0.235,
                          hspace=0.80, wspace=0.26)
    axs_a = [fig.add_subplot(gs[0, j]) for j in range(3)]
    axs_b = [fig.add_subplot(gs[1, j]) for j in range(3)]

    for j, n in enumerate(io.SCALES):
        ax = axs_a[j]
        for method in SERIES_A:
            d = get(t, n, TRACE_KEY[method])
            m = style.METHOD[method]
            ax.plot(d.budget_frac * 100, d["median"], color=m["color"], ls=m["ls"],
                    lw=style.LW_EMPH if method == "bcrea_band" else style.LW_DATA,
                    zorder=3)
        n_runs = int(get(t, n, TRACE_KEY["bcrea_band"]).n_runs.iloc[0])
        ax.set_title(f"{n} targets  ($n$ = {n_runs})", fontsize=9, pad=5)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.yaxis.set_major_locator(MaxNLocator(4))
        if j == 0:
            ax.set_ylabel("nominal objective (targets)")
        style.panel_tag(ax, f"({chr(ord('a') + j)})", dx=-0.02, dy=1.08)

        ax = axs_b[j]
        for method in SERIES_B:
            d = get(t, n, TRACE_KEY_B[method])
            v = d["median"].to_numpy()
            closed = 1.0 - np.abs(v - v[-1]) / abs(v[0] - v[-1])
            m = style.METHOD[method]
            ax.plot(d.budget_frac * 100, closed * 100, color=m["color"], ls=m["ls"],
                    lw=style.LW_EMPH if method == "bcrea_band" else style.LW_DATA, zorder=3)
        ax.set_xlim(0, 100)
        ax.set_ylim(-2, 104)
        ax.set_xticks([0, 50, 100])
        ax.set_yticks([0, 50, 100])
        ax.set_xlabel("decode budget used (% of $D$)")
        if j == 0:
            ax.set_ylabel("share of own gap closed (%)")
        style.panel_tag(ax, f"({chr(ord('d') + j)})", dx=-0.02, dy=1.08)

    handles_a = [plt.Line2D([], [], color=style.METHOD[k]["color"], ls=style.METHOD[k]["ls"],
                            lw=style.LW_EMPH if k == "bcrea_band" else style.LW_DATA,
                            label=f"{style.METHOD[k]['label']}: nominal objective")
                 for k in SERIES_A]
    fig.legend(handles=handles_a, loc="center", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.585), handlelength=2.0, columnspacing=2.2)
    handles_b = [plt.Line2D([], [], color=style.METHOD[k]["color"], ls=style.METHOD[k]["ls"],
                            lw=style.LW_EMPH if k == "bcrea_band" else style.LW_DATA,
                            label=SERIES_B_LABEL[k])
                 for k in SERIES_B]
    fig.legend(handles=handles_b, loc="lower center", ncol=2, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=2.0, columnspacing=2.6)

    require_matplotlib_panel_alignment(
        fig, json_out=str(style.QA_DIR / "figS3_conv.alignment.json"),
        overlay_svg=str(style.QA_DIR / "figS3_conv.alignment.svg"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=False)
    style.save(fig, "figS3_conv")

    for n in io.SCALES:
        row = []
        for method in SERIES_A:
            d = get(t, n, TRACE_KEY[method])
            row.append(f"{style.METHOD[method]['label']} {d['median'].iloc[0]:.2f}"
                       f"->{d['median'].iloc[-1]:.2f}")
        print(f"[figS3] (a) N={n}: " + "; ".join(row))
        for method in SERIES_B:
            d = get(t, n, TRACE_KEY_B[method])
            v = d["median"].to_numpy()
            closed = 1.0 - np.abs(v - v[-1]) / abs(v[0] - v[-1])
            i20 = int(np.searchsorted(d.budget_frac.to_numpy(), 0.25))
            print(f"[figS3] (b) N={n} {method:22s} v0={v[0]:7.3f} vend={v[-1]:7.3f} "
                  f"closed@25%={closed[i20] * 100:5.1f}% max={closed.max() * 100:5.1f}%")
    print(f"[figS3] NSGA-II has no trace series in the batch (excluded from both panels)")


if __name__ == "__main__":
    main()
