# -*- coding: utf-8 -*-
"""TAES 评审轮离线重分析（2026-10-30，只读 20261013_v3 与分析层，无新搜索）。

产出（全部落 data/results/20261013_v3_analysis/）：
  A 段（快，纯 CSV 计算）：
    v3_review_saturation.csv      逐实例资源上限推导与名义最优占比（B1）
    v3_review_drill_family.csv    换案演练按 实例族(conc/disp)×规模 分列 + 名义菜单列（B2/B12）
    v3_review_menu_vs_specialist.csv  认证菜单 vs 66 角点专才菜单同口径对照（B3）
    v3_review_estimators.json     11%/83% 估计量复核（B10）
    v3_review_spread_family.csv   rel_spread 按 实例族×规模 分解（B11）
  B 段（重，读 1800 个 scans JSON + 池 npz，脱控制台跑）：
    v3_review_eps.csv             ε 敏感性（Γ=2，ε∈{0,0.01,0.05,0.10} 的 m* 分布）（B8）
    v3_review_gap_check.csv       MILP 默认 gap vs gap=0 一致性抽查（B9）
    v3_review_fielded_nominal.csv 各策略"上场方案"名义质量（band/sampled/strict 菜单逐模式）（B6）

跑法（仓库根）：
  py -3.11 -X utf8 analysis/analyze_v3_review.py --stage a
  py -3.11 -X utf8 analysis/analyze_v3_review.py --stage b
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))   # 项目根（core 包）

from analyze_v3 import (  # noqa: E402
    DEFAULT_OUT, METHODS, SCAN_DIR, SEEDS, build_instance_order, inst_meta,
)
from analyze_v3_mstar import exact_mstar  # noqa: E402
import core.robust_metrics as r6  # noqa: E402

OUT = Path(DEFAULT_OUT)
POOL_DIR = SCAN_DIR.parent / "pools"
EPS_GRID = (0.0, 0.01, 0.05, 0.10)
DELTA, GAMMA = 0.3, 2
MENU_M = 5


def load(name: str) -> list[dict]:
    with open(OUT / (name + ".csv"), newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for r in rows:
            w.writerow([r.get(k, "") for k in fields])
    print(f"写 {path.name}（{len(rows)} 行）", flush=True)


def med(xs) -> float:
    a = [x for x in xs if x is not None and np.isfinite(x)]
    return float(statistics.median(a)) if a else float("nan")


def resource_cap(n: int, rounds_total: int = 90, kmax: int = 3, p: float = 0.8) -> tuple[float, list[int]]:
    """名义目标上界：把 rounds_total 发弹分配到 n 个目标（每目标 ≤kmax 发，单发命中 ≤p），
    最大化 Σ[1-(1-p)^r_j]。边际递减，贪心即最优。返回 (上界, 分配方案计数)。"""
    alloc = [0] * n
    gains = [1 - (1 - p) ** 1]  # 每目标第一发的边际
    # 简单贪心：每次把一发给边际最大的目标
    marg = [1 - (1 - p) ** 1] * n
    used = 0
    while used < rounds_total:
        j = int(np.argmax(marg))
        if marg[j] <= 0:
            break
        alloc[j] += 1
        used += 1
        r = alloc[j]
        marg[j] = (1 - (1 - p) ** (r + 1)) - (1 - (1 - p) ** r) if r < kmax else -1.0
    val = sum(1 - (1 - p) ** r for r in alloc)
    from collections import Counter
    return val, dict(sorted(Counter(alloc).items()))


# ---------------------------------------------------------------- A 段
def stage_a() -> None:
    t0 = time.time()
    inst_order = build_instance_order()

    # ---- B1 饱和表
    nom = load("v3_nominal")
    best_by_inst: dict[str, list[float]] = {}
    for r in nom:
        if r["method"] == "nominal":
            best_by_inst.setdefault(r["instance"], []).append(float(r["best_nominal"]))
    cap_cache = {}
    rows = []
    for inst in inst_order:
        n, fam, spread = inst_meta(inst)
        if n not in cap_cache:
            cap_cache[n] = resource_cap(n)
        cap, alloc = cap_cache[n]
        best = max(best_by_inst.get(inst, [float("nan")]))
        rows.append({
            "instance": inst, "n_target": n, "family": fam, "spread": spread,
            "upper_bound": round(cap, 4), "alloc": json.dumps(alloc),
            "nominal_best_max": round(best, 4),
            "ratio_best_over_bound": round(best / cap, 6),
            "at_bound": int(best >= cap - 1e-6),
        })
    write_rows(OUT / "v3_review_saturation.csv",
               ["instance", "n_target", "family", "spread", "upper_bound", "alloc",
                "nominal_best_max", "ratio_best_over_bound", "at_bound"], rows)
    for n in (24, 48, 100):
        sub = [r for r in rows if r["n_target"] == n]
        print(f"  {n} tgt: 上界 {sub[0]['upper_bound']}，贴上界实例 {sum(r['at_bound'] for r in sub)}/{len(sub)}，"
              f"最贴 {max(r['ratio_best_over_bound'] for r in sub):.4f}", flush=True)

    # ---- B2/B12 换案分实例族 + 名义菜单
    sw = load("v3_menu_switch")
    rows = []
    for method in ("bcrea_band", "nominal"):
        for n in (24, 48, 100):
            for spread in ("conc", "disp", "all"):
                rr = [r for r in sw if r["method"] == method and r["m"] == str(MENU_M)
                      and r["n_target"] == str(n)
                      and (spread == "all" or r["spread"] == spread)]
                if not rr:
                    continue
                units: dict[tuple[str, str], list[dict]] = {}
                for r in rr:
                    units.setdefault((r["instance"], r["seed"]), []).append(r)
                mean_sw, mean_ro, mean_ns, diffs, ties, below_ns = [], [], [], [], [], []
                for k, ul in units.items():
                    mean_sw.append(np.mean([float(x["switch_qual"]) for x in ul]))
                    mean_ro.append(np.mean([float(x["reopt_best"]) for x in ul]))
                    mean_ns.append(np.mean([float(x["nominal_stick_qual"]) for x in ul]))
                    d = np.array([float(x["reopt_minus_switch"]) for x in ul])
                    diffs.extend(d.tolist())
                    ties.append(float(np.mean(np.abs(d) < 1e-9)))
                    gn = np.array([float(x["gain_switch_vs_nominal_stick"]) for x in ul])
                    below_ns.append(float(np.mean(gn < 0)))
                rows.append({
                    "method": method, "m": MENU_M, "n_target": n, "spread": spread,
                    "n_units": len(units),
                    "mean_switch_median": round(med(mean_sw), 4),
                    "mean_reopt_median": round(med(mean_ro), 4),
                    "mean_nominal_stick_median": round(med(mean_ns), 4),
                    "diff_reopt_minus_switch_median": round(med(diffs), 4),
                    "diff_reopt_minus_switch_max": round(max(diffs), 4),
                    "tie_share_median": round(med(ties), 4),
                    "below_nominal_stick_share_median": round(med(below_ns), 4),
                })
    write_rows(OUT / "v3_review_drill_family.csv",
               ["method", "m", "n_target", "spread", "n_units", "mean_switch_median",
                "mean_reopt_median", "mean_nominal_stick_median",
                "diff_reopt_minus_switch_median", "diff_reopt_minus_switch_max",
                "tie_share_median", "below_nominal_stick_share_median"], rows)
    for r in rows:
        if r["method"] == "bcrea_band":
            print(f"  drill band {r['n_target']} tgt {r['spread']}: 打平率中位 {r['tie_share_median']:.3f} "
                  f"差中位 {r['diff_reopt_minus_switch_median']:+.3f} 最大 {r['diff_reopt_minus_switch_max']:+.3f}", flush=True)

    # ---- B3 认证菜单 vs 66 角点专才菜单（同 6 实例 5 种子）
    rm = load("v3_reopt_menu")
    rc = load("v3_rc")
    drill_insts = sorted({r["instance"] for r in rm})
    rc5_band = {(r["instance"], r["seed"]): float(r["rc5"]) for r in rc
                if r["method"] == "bcrea_band" and r["instance"] in drill_insts}
    rc5_nom = {(r["instance"], r["seed"]): float(r["rc5"]) for r in rc
               if r["method"] == "nominal" and r["instance"] in drill_insts}
    rows = []
    for n in (24, 48, 100):
        rr = [r for r in rm if r["n_target"] == str(n)]
        win_b = tie_b = 0
        win_n = tie_n = 0
        for r in rr:
            key = (r["instance"], r["seed"])
            sp = float(r["rc_menu"])
            if key in rc5_band:
                win_b += int(rc5_band[key] > sp + 1e-9)
                tie_b += int(abs(rc5_band[key] - sp) <= 1e-9)
            if key in rc5_nom:
                win_n += int(rc5_nom[key] > sp + 1e-9)
                tie_n += int(abs(rc5_nom[key] - sp) <= 1e-9)
        rows.append({
            "n_target": n, "n_units": len(rr),
            "rc5_band_median": round(med([rc5_band[(r["instance"], r["seed"])] for r in rr]), 4),
            "rc5_nominal_median": round(med([rc5_nom[(r["instance"], r["seed"])] for r in rr]), 4),
            "specialist_menu_rc_median": round(med([float(r["rc_menu"]) for r in rr]), 4),
            "band_pool_full_rc_median": round(med([float(r["rc_band_pool"]) for r in rr]), 4),
            "specialist_cost_note": "66xD",
            "band_beats_specialist": f"{win_b}/{len(rr)}", "band_ties_specialist": f"{tie_b}/{len(rr)}",
            "nominal_beats_specialist": f"{win_n}/{len(rr)}",
        })
    write_rows(OUT / "v3_review_menu_vs_specialist.csv",
               ["n_target", "n_units", "rc5_band_median", "rc5_nominal_median",
                "specialist_menu_rc_median", "band_pool_full_rc_median",
                "specialist_cost_note", "band_beats_specialist", "band_ties_specialist",
                "nominal_beats_specialist"], rows)
    for r in rows:
        print(f"  专才对照 {r['n_target']} tgt: band菜单RC5 {r['rc5_band_median']} vs 专才 {r['specialist_menu_rc_median']}"
              f"（band 胜 {r['band_beats_specialist']}，平 {r['band_ties_specialist']}）", flush=True)

    # ---- B10 估计量复核（11% / 83%）
    rc_all = load("v3_rc")
    deg = load("v3_degradation")
    def unit_map(rows_in, meth, key):
        return {(r["instance"], r["seed"]): float(r[key]) for r in rows_in if r["method"] == meth}
    est = {}
    for n in (24, 48, 100, "all"):
        rel, ab = [], []
        degrel, degab = [], []
        for r in rc_all:
            if r["method"] != "bcrea_band" or (n != "all" and r["n_target"] != str(n)):
                continue
            k = (r["instance"], r["seed"])
            nv = unit_map(rc_all, "nominal", "rc5").get(k)
            if nv and nv > 0:
                rel.append((float(r["rc5"]) - nv) / nv)
                ab.append(float(r["rc5"]) - nv)
        for r in deg:
            if r["method"] != "bcrea_band" or (n != "all" and r["n_target"] != str(n)):
                continue
            k = (r["instance"], r["seed"])
            nv = unit_map(deg, "nominal", "deg_in_band_mean").get(k)
            if nv and nv > 0:
                degrel.append((nv - float(r["deg_in_band_mean"])) / nv)
                degab.append(nv - float(r["deg_in_band_mean"]))
        est[str(n)] = {
            "rc5_relative_gain_median_pct": round(med(rel) * 100, 2),
            "rc5_absolute_gain_median": round(med(ab), 3),
            "deg_relative_drop_median_pct": round(med(degrel) * 100, 2),
            "deg_absolute_drop_median": round(med(degab), 3),
            "n_units": len(rel),
        }
        print(f"  估计量 {n}: RC5 相对增益中位 {est[str(n)]['rc5_relative_gain_median_pct']}% "
              f"(绝对 {est[str(n)]['rc5_absolute_gain_median']})；退化相对下降中位 "
              f"{est[str(n)]['deg_relative_drop_median_pct']}% (绝对 {est[str(n)]['deg_absolute_drop_median']})", flush=True)
    (OUT / "v3_review_estimators.json").write_text(json.dumps(est, indent=1), encoding="utf-8")
    print("写 v3_review_estimators.json", flush=True)

    # ---- B11 rel_spread 按实例族分解
    ph = load("v3_phenomenon")
    rows = []
    for n in (24, 48, 100):
        for spread in ("conc", "disp", "all"):
            rr = [r for r in ph if r["n_target"] == str(n)
                  and (spread == "all" or r["spread"] == spread)]
            rows.append({
                "n_target": n, "spread": spread, "n_units": len(rr),
                "rel_spread_median_pct": round(med([float(r["rel_spread"]) for r in rr]) * 100, 2),
                "rounds_per_target": round(90 / n, 3),
            })
    write_rows(OUT / "v3_review_spread_family.csv",
               ["n_target", "spread", "n_units", "rel_spread_median_pct", "rounds_per_target"], rows)
    for r in rows:
        print(f"  rel_spread {r['n_target']} tgt {r['spread']}: {r['rel_spread_median_pct']}%", flush=True)
    print(f"A 段完成，用时 {time.time() - t0:.1f}s", flush=True)


# ---------------------------------------------------------------- B 段
def _milp_strict(S: np.ndarray, eps: float) -> int | None:
    """gap=0 的集合覆盖 MILP，与 exact_mstar 同构。"""
    from scipy.optimize import Bounds, LinearConstraint, milp
    rc_full = r6.robust_coverage(S, np.arange(S.shape[1]))
    A = (S >= rc_full - eps).astype(float)
    res = milp(np.ones(S.shape[1]), integrality=np.ones(S.shape[1]),
               bounds=Bounds(0, 1), constraints=LinearConstraint(A, 1, np.inf),
               options={"mip_rel_gap": 0.0})
    return int(round(float(res.fun))) if res.success else None


def stage_b() -> None:
    t0 = time.time()
    inst_order = build_instance_order()
    rng = random.Random(20261030)

    # ---- B8 ε 敏感性（Γ=2）+ B9 gap 抽查
    eps_rows, gap_rows = [], []
    gap_units = []
    for n in (24, 48, 100):
        for m in METHODS:
            for i in inst_order:
                if inst_meta(i)[0] != n:
                    continue
                gap_units.append((m, i, rng.choice(SEEDS)))
    gap_set = set(gap_units)
    acc: dict[tuple, dict] = {}
    for mi, method in enumerate(METHODS):
        for inst in inst_order:
            n, fam, spread = inst_meta(inst)
            for s in SEEDS:
                fp = SCAN_DIR / f"{method}_{inst}_s{s:02d}.json"
                js = json.loads(fp.read_text(encoding="utf-8"))
                eq = np.asarray(js["nominal"]["eq_idx"], int)
                S = np.asarray(js["cells"][f"d{int(DELTA * 100)}_g{GAMMA}"]["objsS"], float)[:, eq]
                per_eps = {}
                for eps in EPS_GRID:
                    ex, gr, rc, ok = exact_mstar(S, eps)
                    per_eps[eps] = ex
                key = (n, method)
                for eps, ex in per_eps.items():
                    a = acc.setdefault((n, method, eps), {"mstars": [], "ok": 0})
                    if ex is not None:
                        a["mstars"].append(ex)
                        a["ok"] += 1
                if (method, inst, s) in gap_set:
                    strict = _milp_strict(S, 0.01)
                    gap_rows.append({
                        "method": method, "instance": inst, "seed": s, "n_target": n,
                        "eq_size": int(len(eq)),
                        "mstar_default": per_eps[0.01], "mstar_gap0": strict,
                        "agree": int(strict == per_eps[0.01]),
                    })
        print(f"  [{mi + 1}/{len(METHODS)}] {method} 完（{time.time() - t0:.1f}s）", flush=True)
    for (n, method, eps), a in sorted(acc.items()):
        eps_rows.append({
            "n_target": n, "method": method, "gamma": GAMMA, "eps": eps,
            "n_units": len(a["mstars"]),
            "mstar_median": med(a["mstars"]),
            "share_mstar_1": round(float(np.mean([v == 1 for v in a["mstars"]])), 4) if a["mstars"] else "",
            "mstar_max": max(a["mstars"]) if a["mstars"] else "",
        })
    write_rows(OUT / "v3_review_eps.csv",
               ["n_target", "method", "gamma", "eps", "n_units", "mstar_median",
                "share_mstar_1", "mstar_max"], eps_rows)
    write_rows(OUT / "v3_review_gap_check.csv",
               ["method", "instance", "seed", "n_target", "eq_size",
                "mstar_default", "mstar_gap0", "agree"], gap_rows)
    n_agree = sum(r["agree"] for r in gap_rows)
    print(f"  gap 抽查：默认与 gap=0 一致 {n_agree}/{len(gap_rows)}", flush=True)

    # ---- B6 上场方案名义质量（band / strict / sampled 的 m=5 菜单逐模式上场成员）
    nom = load("v3_nominal")
    ref = {}
    for r in nom:
        if r["method"] == "nominal":
            ref[r["instance"]] = max(ref.get(r["instance"], -np.inf), float(r["best_nominal"]))
    field_rows = []
    for method in ("bcrea_band", "bcrea_strict", "sampled_wc"):
        accf: dict[tuple, dict] = {}
        for inst in inst_order:
            n, fam, spread = inst_meta(inst)
            for s in SEEDS:
                sj = SCAN_DIR / f"{method}_{inst}_s{s:02d}.json"
                pj = POOL_DIR / f"{method}_{inst}_s{s:02d}.npz"
                if not sj.exists() or not pj.exists():
                    continue
                js = json.loads(sj.read_text(encoding="utf-8"))
                with np.load(pj, allow_pickle=False) as z:
                    decs = np.asarray(z["decs"])  # 与 analyze_v3_menu_switch.load_decs 同口径
                menu = r6.extract_menu(js, decs, m=MENU_M, delta=DELTA, gamma=GAMMA)
                members = np.asarray(menu["members"], int)
                S = np.asarray(js["cells"][f"d{int(DELTA * 100)}_g{GAMMA}"]["objsS"], float)
                objs0 = np.asarray(js["nominal"]["objs0"], float)
                sel = S[:, members].argmax(axis=1)          # 每模式上场的成员
                fielded_f0 = objs0[members[sel]]            # 上场方案的名义值
                rel_gap = (ref[inst] - fielded_f0) / ref[inst] * 100.0
                pool_best = float(objs0.max())
                a = accf.setdefault((n, method), {"rel": [], "pool_gap": [], "sizes": []})
                a["rel"].extend(rel_gap.tolist())
                a["pool_gap"].append((ref[inst] - pool_best) / ref[inst] * 100.0)
                a["sizes"].append(len(members))
        for (n, m), a in sorted(accf.items()):
            field_rows.append({
                "n_target": n, "method": m, "n_units": len(a["pool_gap"]),
                "fielded_nominal_gap_median_pct": round(med(a["rel"]), 3),
                "fielded_nominal_gap_p90_pct": round(float(np.percentile(a["rel"], 90)), 3),
                "pool_best_gap_median_pct": round(med(a["pool_gap"]), 3),
                "menu_size_median": med(a["sizes"]),
            })
    write_rows(OUT / "v3_review_fielded_nominal.csv",
               ["n_target", "method", "n_units", "fielded_nominal_gap_median_pct",
                "fielded_nominal_gap_p90_pct", "pool_best_gap_median_pct",
                "menu_size_median"], field_rows)
    for r in field_rows:
        print(f"  上场名义 {r['n_target']} tgt {r['method']}: 中位 {r['fielded_nominal_gap_median_pct']}% "
              f"p90 {r['fielded_nominal_gap_p90_pct']}% 池最好差 {r['pool_best_gap_median_pct']}%", flush=True)
    print(f"B 段完成，用时 {time.time() - t0:.1f}s", flush=True)


# ---------------------------------------------------------------- C 段：识别误差实验
P_GRID = (0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
N_DRAWS = 400
IDENT_RNG = 20261030


def _mode_pairs() -> list[tuple[int, int]]:
    from itertools import combinations
    return list(combinations(range(12), 2))  # 与 scans 的角点枚举同序（对齐已验证）


def stage_c() -> None:
    """B5 识别误差实验：菜单换案在模式误识别下的质量（纯后处理，无新搜索）。

    两个误差模型（都是对"观测模式 û"的扰动，真实模式 u 不变）：
      uniform：以概率 p 把 û 换成其余 65 个模式的均匀抽样（完全错认）；
      shared ：以概率 p 把 û 换成与 u 恰共享一个退化分量的模式（认出一半）。
    上场价值 = S[u, argmax_m S[û, m]]（按观测查表、按真实结算）。
    """
    from itertools import combinations
    t0 = time.time()
    rm = load("v3_reopt_menu")
    units = sorted({(r["instance"], int(r["seed"])) for r in rm})
    pairs = _mode_pairs()
    n_modes = len(pairs)
    # 每个模式的"共享一个分量"误差集
    shared_sets = []
    for (a, b) in pairs:
        sh = [k for k, (x, y) in enumerate(pairs)
              if (x == a or y == b) and not (x == a and y == b)]
        shared_sets.append(np.asarray(sh, int))
    rows = []
    per_unit = {}
    for inst, s in units:
        n, fam, spread = inst_meta(inst)
        sj = SCAN_DIR / f"bcrea_band_{inst}_s{s:02d}.json"
        pj = POOL_DIR / f"bcrea_band_{inst}_s{s:02d}.npz"
        js = json.loads(sj.read_text(encoding="utf-8"))
        with np.load(pj, allow_pickle=False) as z:
            decs = np.asarray(z["decs"])
        menu = r6.extract_menu(js, decs, m=MENU_M, delta=DELTA, gamma=GAMMA)
        members = np.asarray(menu["members"], int)
        S = np.asarray(js["cells"][f"d{int(DELTA * 100)}_g{GAMMA}"]["objsS"], float)[:, members]
        best_member_per_mode = S.argmax(axis=1)          # 查表规则：观测模式下选的成员
        perfect = S.max(axis=1)                          # 完美识别的逐模式价值
        rng = np.random.default_rng(IDENT_RNG + s * 100 + n)
        for model in ("uniform", "shared"):
            for p in P_GRID:
                vals = np.empty(N_DRAWS)
                for d in range(N_DRAWS):
                    obs = np.arange(n_modes)
                    if p > 0:
                        err = rng.random(n_modes) < p
                        for u in np.nonzero(err)[0]:
                            if model == "uniform":
                                cand = rng.integers(0, n_modes - 1)
                                obs[u] = cand if cand < u else cand + 1
                            else:
                                sh = shared_sets[u]
                                obs[u] = sh[rng.integers(0, len(sh))]
                    vals[d] = float(np.mean(S[np.arange(n_modes), best_member_per_mode[obs]]))
                per_unit.setdefault((n, model, p), []).append(float(vals.mean()))
        per_unit.setdefault((n, "perfect", 0.0), []).append(float(perfect.mean()))
        per_unit.setdefault((n, "rc_floor", 0.0), []).append(float(perfect.min()))
    for (n, model, p), v in sorted(per_unit.items()):
        rows.append({
            "n_target": n, "model": model, "p": p, "n_units": len(v),
            "mean_fielded_median": round(med(v), 4),
            "mean_fielded_q25": round(float(np.percentile(v, 25)), 4),
            "mean_fielded_q75": round(float(np.percentile(v, 75)), 4),
        })
    write_rows(OUT / "v3_review_ident.csv",
               ["n_target", "model", "p", "n_units", "mean_fielded_median",
                "mean_fielded_q25", "mean_fielded_q75"], rows)
    for n in (24, 48, 100):
        perf = med(per_unit[(n, "perfect", 0.0)])
        floor = med(per_unit[(n, "rc_floor", 0.0)])
        for model in ("uniform", "shared"):
            cells = []
            for p in (0.1, 0.4, 1.0):
                v = med(per_unit[(n, model, p)])
                cells.append(f"p={p}: {v:.2f} ({(v - floor) / (perf - floor) * 100 if perf > floor else float('nan'):.0f}%)")
            print(f"  识别 {n} tgt {model}: 完美 {perf:.2f} / RC 下限 {floor:.2f} | " + " ".join(cells), flush=True)
    print(f"C 段完成，用时 {time.time() - t0:.1f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("a", "b", "c"), required=True)
    args = ap.parse_args()
    (stage_a if args.stage == "a" else stage_b if args.stage == "b" else stage_c)()


if __name__ == "__main__":
    main()
