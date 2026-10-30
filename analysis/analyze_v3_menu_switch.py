# -*- coding: utf-8 -*-
"""v3 阶段1-B：认证菜单 × 66 角点换案评估（零新搜索、零新解码）。

核心：scans 的 cells.d30_g2.objsS 已是"整池方案 × 66 个 Γ=2 角点场景"的质量矩阵。
  - 换案质量 = 菜单成员列在该角点的 max（观测到角点后从菜单里挑最好的）
  - 不换案基线 = 本池名义最优单解（argmax objs0）在该角点的值
  - 名义池不换案基线 = nominal 池名义最优单解在该角点的值（旧稿 P_N 口径）
  - 重解最优 = 对应角点 reopt npz 的 max(objs)（真实重优化，成本 9000N 解码/角点）
  - 菜单 RC = 66 角点换案质量的最小值

角点对齐（显式对照表，见 CORNERS）：
  scans 的 d30_g2 用 core.robust_metrics.sample_masks(Γ=2, n_comp=12) 确定性穷举，
  顺序 = itertools.combinations(range(12), 2) 的字典序：
  c0c1, c0c2, …, c0c11, c1c2, …, c10c11（共 66 个，0-based 分量索引）。
  reopt 文件名 reopt_c{a}c{b}_… 用同一 combinations 顺序生成（make_tasks_v3.py:124-131）。
  脚本会逐项核对 masks 行 ↔ (a,b)，并用"重解最优解回到本角点重解码"做数值核对
  （应逐位相等；见 v3_corner_alignment_check.csv）。

范围：6 代表实例 × 5 种子 × 6 方法 × m∈{1,3,5} × 66 角点。
输出：
  v3_menu_switch.csv            35640 行逐角点长表
  v3_menu_switch_summary.csv    规模×方法×m 汇总（配对差中位/IQR/胜率）
  v3_menu_switch_summary.txt    人读摘要（含旧稿口径的均值对比）
  v3_corner_alignment.csv       66 角点显式对照表 + 全量核对结果
  v3_corner_alignment_check.csv 数值核对（6 实例 × 66 角点重解码）

跑法（仓库根）：
  py -3.11 -X utf8 analysis/analyze_v3_menu_switch.py
  py -3.11 -X utf8 analysis/analyze_v3_menu_switch.py --limit 1 --skip-align-decode
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))   # 项目根（core 包）

import core.robust_metrics as r6  # noqa: E402
from core.model import DefenseParams, SWTAProblem  # noqa: E402
from core.scenarios import load_enemy  # noqa: E402
from core.decoder import evaluate  # noqa: E402
from analyze_v3 import (  # noqa: E402
    DEFAULT_OUT, METHODS, POOL_DIR, REOPT_DIR, REOPT_INSTANCES, SCAN_DIR,
    SHORT, inst_meta, scenario_path,
)

N_COMP = 12
CORNERS = list(combinations(range(N_COMP), 2))          # 66 个 0-based 分量对（字典序）
CORNER_IDS = [f"c{a}c{b}" for a, b in CORNERS]
MENU_SIZES = [1, 3, 5]
REOPT_SEEDS = list(range(1, 6))     # reopt 存档只有 5 个种子
DELTA, GAMMA = 0.3, 2
MAIN_CELL = "d30_g2"


def fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "1" if v else "0"
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if np.isnan(v):
            return "nan"
        if np.isinf(v):
            return "inf" if v > 0 else "-inf"
        return format(v, ".15g")
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return str(v)


def write_rows(path: Path, fields: list[str], rows: list[dict]) -> int:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for r in rows:
            w.writerow([fmt(r.get(k)) for k in fields])
    return len(rows)


def med(x) -> float:
    a = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    return float(np.median(a)) if len(a) else float("nan")


def q25(x) -> float:
    a = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    return float(np.percentile(a, 25)) if len(a) else float("nan")


def q75(x) -> float:
    a = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    return float(np.percentile(a, 75)) if len(a) else float("nan")


def load_scan(method: str, inst: str, seed: int) -> dict:
    fp = SCAN_DIR / f"{method}_{inst}_s{seed:02d}.json"
    return json.loads(fp.read_text(encoding="utf-8"))


def load_decs(method: str, inst: str, seed: int) -> np.ndarray:
    z = np.load(POOL_DIR / f"{method}_{inst}_s{seed:02d}.npz", allow_pickle=False)
    return np.asarray(z["decs"])


def load_reopt_best(inst: str, seed: int) -> np.ndarray:
    """66 个角点的重解最优（按 CORNERS 顺序）。"""
    vals = np.full(len(CORNERS), np.nan)
    for k, (a, b) in enumerate(CORNERS):
        fp = REOPT_DIR / f"reopt_c{a}c{b}_{inst}_s{seed:02d}.npz"
        z = np.load(fp, allow_pickle=False)
        vals[k] = float(np.asarray(z["objs"], float).max())
    return vals


# ---------------------------------------------------------------- 对齐核对
def check_alignment(full_mask_check: bool) -> tuple[list[dict], bool]:
    """逐项核对 masks 行 ↔ c{a}c{b} ↔ reopt 配置。返回 (行, 全部一致)。"""
    ref = None
    rows = []
    files = sorted(SCAN_DIR.glob("*.json"))
    if not full_mask_check:
        # 每个 (method, 规模) 抽一个实例，覆盖 6 方法 × 3 规模
        keep = {}
        for fp in files:
            for m in METHODS:
                if fp.stem.startswith(m + "_"):
                    rest = fp.stem[len(m) + 1:]
                    inst = rest.rsplit("_s", 1)[0]
                    key = (m, inst_meta(inst)[0])
                    keep.setdefault(key, fp)
                    break
        files = sorted(keep.values())
    mask_ok = True
    for fp in files:
        js = json.loads(fp.read_text(encoding="utf-8"))
        masks = np.asarray(js["cells"][MAIN_CELL]["masks"], np.int8)
        if masks.shape != (len(CORNERS), N_COMP):
            mask_ok = False
            continue
        expect = np.zeros_like(masks)
        for k, (a, b) in enumerate(CORNERS):
            expect[k, a] = 1
            expect[k, b] = 1
        if not np.array_equal(masks, expect):
            mask_ok = False
        if ref is None:
            ref = masks
        elif not np.array_equal(ref, masks):
            mask_ok = False
    for k, (a, b) in enumerate(CORNERS):
        n_have = 0
        cfg_ok = 0
        for inst in REOPT_INSTANCES:
            for s in range(1, 6):
                fp = REOPT_DIR / f"reopt_c{a}c{b}_{inst}_s{s:02d}.npz"
                if not fp.exists():
                    continue
                n_have += 1
                z = np.load(fp, allow_pickle=False)
                cfg = json.loads(str(z["config"]))
                if (list(cfg.get("components") or []) == [a, b]
                        and int(cfg.get("gamma", -1)) == GAMMA
                        and abs(float(cfg.get("delta", -1)) - DELTA) < 1e-12):
                    cfg_ok += 1
        rows.append({
            "corner_idx": k, "corner": CORNER_IDS[k], "comp_a": a, "comp_b": b,
            "mask_sum": 2, "mask_matches_pair": True,
            "reopt_files": n_have, "reopt_config_ok": cfg_ok,
            "scans_checked": len(files), "masks_identical_across_scans": mask_ok,
        })
    return rows, mask_ok


def check_alignment_decode(instances: list[str], seed: int, backend: str) -> list[dict]:
    """数值核对：reopt 最优解回到扫描构造的同一角点问题重解码，应等于 reopt max(objs)。"""
    rows = []
    for inst in instances:
        js = load_scan("nominal", inst, seed)
        masks = np.asarray(js["cells"][MAIN_CELL]["masks"], np.int8)
        enemy = load_enemy(str(ROOT / scenario_path(inst)))
        base = DefenseParams()
        n_r = base.n_radar
        for k, (a, b) in enumerate(CORNERS):
            fp = REOPT_DIR / f"reopt_c{a}c{b}_{inst}_s{seed:02d}.npz"
            z = np.load(fp, allow_pickle=False)
            objs = np.asarray(z["objs"], float)
            decs = np.asarray(z["decs"])
            d = decs[int(np.argmax(objs))]
            rad, mis = r6.scales_from_mask(masks[k], DELTA, n_r)
            prob = SWTAProblem(enemy, r6.perturbed_params(base, rad, mis))
            if backend == "gpu":
                v = float(r6.eval_pool(prob, d[None, :], backend="gpu")[0])
            else:
                v = float(evaluate(prob, d, infeasible_ok=True).obj)
            rows.append({
                "instance": inst, "seed": seed, "corner_idx": k, "corner": CORNER_IDS[k],
                "comp_a": a, "comp_b": b, "reopt_best": float(objs.max()),
                "redecoded_best": v, "abs_diff": abs(v - float(objs.max())),
            })
        print(f"  [align] {inst}: 66 角点重解码最大差 "
              f"{max(r['abs_diff'] for r in rows[-66:]):.3e}", flush=True)
    return rows


# ---------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description="认证菜单 × 66 角点换案评估（复用已存质量矩阵）")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=0, help="冒烟：只用前 N 个代表实例")
    ap.add_argument("--skip-align-decode", action="store_true", help="跳过数值对齐核对（省 1 分钟）")
    ap.add_argument("--sample-mask-check", action="store_true",
                    help="只抽查 18 个 scan 的 masks（默认解析全部 1800 个）")
    ap.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    instances = REOPT_INSTANCES[: args.limit] if args.limit else REOPT_INSTANCES
    print(f"实例 {instances}；m∈{MENU_SIZES}；方法 {METHODS}", flush=True)

    # ---------- 1. 角点对齐
    align_rows, mask_ok = check_alignment(not args.sample_mask_check)
    n_files = sum(r["reopt_files"] for r in align_rows)
    n_cfg = sum(r["reopt_config_ok"] for r in align_rows)
    print(f"对齐：masks 全一致={mask_ok}，reopt 文件 {n_files}/1980，配置一致 {n_cfg}/1980",
          flush=True)
    write_rows(out / "v3_corner_alignment.csv",
               ["corner_idx", "corner", "comp_a", "comp_b", "mask_sum", "mask_matches_pair",
                "reopt_files", "reopt_config_ok", "scans_checked",
                "masks_identical_across_scans"], align_rows)
    if not args.skip_align_decode:
        dec_rows = check_alignment_decode(instances, 1, args.backend)
        write_rows(out / "v3_corner_alignment_check.csv",
                   ["instance", "seed", "corner_idx", "corner", "comp_a", "comp_b",
                    "reopt_best", "redecoded_best", "abs_diff"], dec_rows)
        print(f"数值对齐：{len(dec_rows)} 条，最大差 "
              f"{max(r['abs_diff'] for r in dec_rows):.3e}", flush=True)

    # ---------- 2. 逐角点换案评估
    rows: list[dict] = []
    nominal_cache: dict[tuple[str, int], tuple[np.ndarray, int]] = {}
    reopt_cache: dict[tuple[str, int], np.ndarray] = {}
    for inst in instances:
        n, fam, spread = inst_meta(inst)
        for s in REOPT_SEEDS:
            if (inst, s) not in reopt_cache:
                reopt_cache[(inst, s)] = load_reopt_best(inst, s)
            reopt_best = reopt_cache[(inst, s)]
            # 名义池不换案基线（旧稿 P_N）：nominal 池名义最优单解
            if (inst, s) not in nominal_cache:
                js_n = load_scan("nominal", inst, s)
                S_n = np.asarray(js_n["cells"][MAIN_CELL]["objsS"], float)
                b_n = int(np.argmax(np.asarray(js_n["nominal"]["objs0"], float)))
                nominal_cache[(inst, s)] = (S_n[:, b_n].copy(), b_n)
            nom_stick, _ = nominal_cache[(inst, s)]
            for method in METHODS:
                js = load_scan(method, inst, s)
                objs0 = np.asarray(js["nominal"]["objs0"], float)
                S = np.asarray(js["cells"][MAIN_CELL]["objsS"], float)
                b_idx = int(np.argmax(objs0))
                stick = S[:, b_idx]
                decs = load_decs(method, inst, s)
                for m in MENU_SIZES:
                    menu = r6.extract_menu(js, decs, m=m, delta=DELTA, gamma=GAMMA)
                    members = np.asarray(menu["members"], int)
                    sw = S[:, members].max(axis=1)
                    menu_rc = float(sw.min())
                    for k, (a, b) in enumerate(CORNERS):
                        rows.append({
                            "instance": inst, "family": fam, "spread": spread,
                            "n_target": n, "seed": s, "method": method, "m": m,
                            "corner_idx": k, "corner": CORNER_IDS[k],
                            "comp_a": a, "comp_b": b,
                            "switch_qual": float(sw[k]),
                            "stick_qual": float(stick[k]),
                            "nominal_stick_qual": float(nom_stick[k]),
                            "reopt_best": float(reopt_best[k]),
                            "gain_switch_vs_stick": float(sw[k] - stick[k]),
                            "gain_switch_vs_nominal_stick": float(sw[k] - nom_stick[k]),
                            "reopt_minus_switch": float(reopt_best[k] - sw[k]),
                            "menu_rc": menu_rc,
                            "stick_wc": float(stick.min()),
                            "menu_size": int(len(members)),
                        })
        print(f"  [{inst}] 完（{time.time() - t_start:.1f}s，累计 {len(rows)} 行）", flush=True)

    fields = ["instance", "family", "spread", "n_target", "seed", "method", "m",
              "corner_idx", "corner", "comp_a", "comp_b", "switch_qual", "stick_qual",
              "nominal_stick_qual", "reopt_best", "gain_switch_vs_stick",
              "gain_switch_vs_nominal_stick", "reopt_minus_switch", "menu_rc",
              "stick_wc", "menu_size"]
    n1 = write_rows(out / "v3_menu_switch.csv", fields, rows)
    print(f"v3_menu_switch.csv 写 {n1} 行", flush=True)

    # ---------- 3. 汇总
    sum_rows = []
    for n in (24, 48, 100):
        for method in METHODS:
            for m in MENU_SIZES:
                rr = [r for r in rows if r["n_target"] == n and r["method"] == method
                      and r["m"] == m]
                if not rr:
                    continue
                units = defaultdict(list)
                for r in rr:
                    units[(r["instance"], r["seed"])].append(r)
                mean_sw = [float(np.mean([x["switch_qual"] for x in v])) for v in units.values()]
                mean_st = [float(np.mean([x["stick_qual"] for x in v])) for v in units.values()]
                mean_ns = [float(np.mean([x["nominal_stick_qual"] for x in v]))
                           for v in units.values()]
                mean_ro = [float(np.mean([x["reopt_best"] for x in v])) for v in units.values()]
                d_ro = [x["reopt_minus_switch"] for x in rr]
                d_st = [x["gain_switch_vs_stick"] for x in rr]
                d_ns = [x["gain_switch_vs_nominal_stick"] for x in rr]
                sum_rows.append({
                    "n_target": n, "method": method, "m": m,
                    "n_units": len(units), "n_corner_pairs": len(rr),
                    "mean_switch_median": med(mean_sw),
                    "mean_switch_q25": q25(mean_sw), "mean_switch_q75": q75(mean_sw),
                    "mean_stick_median": med(mean_st),
                    "mean_nominal_stick_median": med(mean_ns),
                    "mean_reopt_median": med(mean_ro),
                    "mean_reopt_q25": q25(mean_ro), "mean_reopt_q75": q75(mean_ro),
                    "diff_reopt_minus_switch_median": med(d_ro),
                    "diff_reopt_minus_switch_q25": q25(d_ro),
                    "diff_reopt_minus_switch_q75": q75(d_ro),
                    "share_switch_beats_reopt": float(np.mean(np.asarray(d_ro) < -1e-9)),
                    "share_switch_ties_reopt": float(np.mean(np.abs(np.asarray(d_ro)) <= 1e-9)),
                    "diff_switch_minus_stick_median": med(d_st),
                    "diff_switch_minus_stick_q25": q25(d_st),
                    "diff_switch_minus_stick_q75": q75(d_st),
                    "share_gain_vs_stick_positive": float(np.mean(np.asarray(d_st) > 1e-9)),
                    "share_gain_vs_stick_negative": float(np.mean(np.asarray(d_st) < -1e-9)),
                    "diff_switch_minus_nominal_stick_median": med(d_ns),
                    "share_gain_vs_nominal_positive": float(np.mean(np.asarray(d_ns) > 1e-9)),
                    "menu_rc_median": med([x["menu_rc"] for x in rr]),
                    "stick_wc_median": med([x["stick_wc"] for x in rr]),
                    "mean_switch_over_reopt_median":
                        med([a / b if b else float("nan") for a, b in zip(mean_sw, mean_ro)]),
                })
    fields_sum = ["n_target", "method", "m", "n_units", "n_corner_pairs",
                  "mean_switch_median", "mean_switch_q25", "mean_switch_q75",
                  "mean_stick_median", "mean_nominal_stick_median", "mean_reopt_median",
                  "mean_reopt_q25", "mean_reopt_q75", "diff_reopt_minus_switch_median",
                  "diff_reopt_minus_switch_q25", "diff_reopt_minus_switch_q75",
                  "share_switch_beats_reopt", "share_switch_ties_reopt",
                  "diff_switch_minus_stick_median", "diff_switch_minus_stick_q25",
                  "diff_switch_minus_stick_q75", "share_gain_vs_stick_positive",
                  "share_gain_vs_stick_negative", "diff_switch_minus_nominal_stick_median",
                  "share_gain_vs_nominal_positive", "menu_rc_median", "stick_wc_median",
                  "mean_switch_over_reopt_median"]
    n2 = write_rows(out / "v3_menu_switch_summary.csv", fields_sum, sum_rows)
    print(f"v3_menu_switch_summary.csv 写 {n2} 行", flush=True)

    # ---------- 4. 人读摘要（旧稿口径：66 角点质量均值，逐实例）
    lines = ["v3 认证菜单 × 66 角点换案评估（δ=0.3, Γ=2；复用 scans 质量矩阵，零新搜索）", ""]
    lines.append(f"角点对齐：masks 与 combinations(range(12),2) 字典序全一致={mask_ok}；"
                 f"reopt 文件 {n_files}/1980、配置一致 {n_cfg}/1980")
    lines.append("角点顺序（显式）：c0c1, c0c2, …, c0c11, c1c2, …, c10c11；0-based 分量索引")
    lines.append("")
    lines.append("== 旧稿同口径（66 角点质量均值；每实例取 5 种子中位，再给 6 实例范围）")
    lines.append("方法             m  换案均值(中位/范围)        重解均值(中位)   不换案(本池)   名义池不换案")
    for method in METHODS:
        for m in MENU_SIZES:
            vals_sw, vals_ro, vals_st, vals_ns = [], [], [], []
            for inst in instances:
                rr = [r for r in rows if r["instance"] == inst and r["method"] == method
                      and r["m"] == m]
                if not rr:
                    continue
                seeds = sorted({r["seed"] for r in rr})
                per_sw = [np.mean([x["switch_qual"] for x in rr if x["seed"] == sd])
                          for sd in seeds]
                per_ro = [np.mean([x["reopt_best"] for x in rr if x["seed"] == sd])
                          for sd in seeds]
                per_st = [np.mean([x["stick_qual"] for x in rr if x["seed"] == sd])
                          for sd in seeds]
                per_ns = [np.mean([x["nominal_stick_qual"] for x in rr if x["seed"] == sd])
                          for sd in seeds]
                vals_sw.append(med(per_sw))
                vals_ro.append(med(per_ro))
                vals_st.append(med(per_st))
                vals_ns.append(med(per_ns))
            if not vals_sw:
                continue
            lines.append(f"{SHORT[method]:14s} {m:>2}  "
                         f"{med(vals_sw):8.3f} [{min(vals_sw):.3f},{max(vals_sw):.3f}]   "
                         f"{med(vals_ro):8.3f}      {med(vals_st):8.3f}      {med(vals_ns):8.3f}")
    lines.append("")
    lines.append("== 逐角点配对差（全 66×30 配对池化；中位 [Q25,Q75] / 胜率）")
    lines.append("规模 方法          m  重解−换案              换案−本池不换案        换案−名义池不换案")
    for r in sum_rows:
        lines.append(f"{r['n_target']:>4} {SHORT[r['method']]:12s} {r['m']:>2}  "
                     f"{r['diff_reopt_minus_switch_median']:+.3f} "
                     f"[{r['diff_reopt_minus_switch_q25']:+.3f},"
                     f"{r['diff_reopt_minus_switch_q75']:+.3f}] "
                     f"胜{r['share_switch_beats_reopt']:.2f}  "
                     f"{r['diff_switch_minus_stick_median']:+.3f} "
                     f"胜{r['share_gain_vs_stick_positive']:.2f}  "
                     f"{r['diff_switch_minus_nominal_stick_median']:+.3f} "
                     f"胜{r['share_gain_vs_nominal_positive']:.2f}")
    lines.append("")
    lines.append("注：换案=每角点取菜单成员在该角点的最好值（观测到角点后切换）；"
                 "重解=该角点独立重优化池的最优（66×9000N 解码）；")
    lines.append("    本池不换案=该池名义最优单解；名义池不换案=nominal 池名义最优单解（旧稿 P_N）。")
    lines.append("    '胜' 指换案严格优于对照的角点比例（差>1e-9）。")
    (out / "v3_menu_switch_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:8]), flush=True)
    print(f"全部完成，用时 {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
