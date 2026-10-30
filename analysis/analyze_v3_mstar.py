# -*- coding: utf-8 -*-
"""v3 阶段1-A：m* 精确覆盖数重算（只读 20261013_v3，不跑搜索）。

对全部 1800 个解池 × δ=0.3 主档 × Γ∈{1,2,3,12}（scans 的 d30_g1/g2/g3/g12 四格）：
  - exact_mstar：MILP（scipy.optimize.milp / HiGHS）求把整池 RC 抬到 rc_full−eps 所需的最少方案数
  - greedy_rc_select 的停止数（贪心要几个才达标）
  - rc_full / eps / 等价带规模
口径与 analyze_hardfix.exact_mstar 完全一致（eps=0.01，S 限定在名义等价带 eq_idx 上）。
  m* 是"认证菜单最少要几个方案"的精确值；贪心停止数是启发式菜单大小。

输出（data/results/20261013_v3_analysis/）：
  v3_mstar.csv        1800×4 = 7200 行，逐池逐 Γ
  v3_mstar_dist.csv   规模×方法×Γ 的 m* / 贪心 分布计数
  v3_mstar_summary.txt 人读摘要（中位数表 + 塌为 1 的比例）
  v3_mstar_brute_check.csv  随机 5 个小池的暴力枚举核对

跑法（仓库根）：
  py -3.11 -X utf8 analysis/analyze_v3_mstar.py
  py -3.11 -X utf8 analysis/analyze_v3_mstar.py --limit 2      # 冒烟
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))   # 项目根（core 包）

from core.robust_metrics import robust_coverage  # noqa: E402
from analyze_v3 import (  # noqa: E402
    DEFAULT_OUT, METHODS, SCAN_DIR, SEEDS, SHORT, build_instance_order, inst_meta,
)

GAMMAS = [1, 2, 3, 12]
DELTA = 0.3
EPS = 0.01
BRUTE_N = 5
BRUTE_CAP = 4
HIST_M = 12     # m* 直方图列 1..12
HIST_G = 20     # 贪心直方图列 1..20，>20 汇总


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


def greedy_stop(S: np.ndarray, thr: float) -> int:
    """贪心顺序（与 core.robust_metrics.greedy_rc_select 逐步一致）达到 thr 所需个数；提前停。"""
    n = S.shape[1]
    chosen = np.zeros(n, dtype=bool)
    cur_max = np.full(S.shape[0], -np.inf)
    for k in range(1, n + 1):
        rc = np.maximum(cur_max[:, None], S).min(axis=0)
        rc[chosen] = -np.inf
        i = int(rc.argmax())
        chosen[i] = True
        cur_max = np.maximum(cur_max, S[:, i])
        if float(cur_max.min()) >= thr:
            return k
    return n


def exact_mstar(S: np.ndarray, eps: float = EPS) -> tuple[int | None, int, float, bool]:
    """S: (n_scen, n_plans)。返回 (exact, greedy_stop, rc_full, milp_ok)。

    与 analyze_hardfix.exact_mstar 同口径（MILP + 贪心停止数），贪心改为达标即停，
    结果与"跑满整池再找停止点"逐点一致。
    """
    rc_full = robust_coverage(S, np.arange(S.shape[1]))
    t = rc_full - eps
    A = (S >= t).astype(float)
    res = milp(np.ones(S.shape[1]), integrality=np.ones(S.shape[1]),
               bounds=Bounds(0, 1), constraints=LinearConstraint(A, 1, np.inf))
    ok = bool(res.success)
    exact = int(round(float(res.fun))) if ok else None
    g = greedy_stop(S, t)
    return exact, g, rc_full, ok


def brute_mstar(S: np.ndarray, eps: float = EPS, cap: int = BRUTE_CAP) -> tuple[int | None, int]:
    """穷举 ≤cap 的子集求精确 m*；返回 (m* 或 None 表示 >cap, 检查的子集数)。"""
    n_scen, n_pool = S.shape
    rc_full = float(S.max(axis=1).min())
    thr = rc_full - eps
    n_checked = 0
    for m in range(1, min(cap, n_pool) + 1):
        for idx in combinations(range(n_pool), m):
            n_checked += 1
            if S[:, list(idx)].max(axis=1).min() >= thr:
                return m, n_checked
    return None, n_checked


def main() -> None:
    ap = argparse.ArgumentParser(description="v3 精确覆盖数 m* 重算")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=0, help="冒烟：只取实例顺序前 N 个")
    ap.add_argument("--brute-seed", type=int, default=20261019)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    inst_order = build_instance_order()
    if args.limit:
        inst_order = inst_order[: args.limit]
    units = [(m, i, s) for m in METHODS for i in inst_order for s in SEEDS]
    print(f"单元 {len(units)} × Γ {len(GAMMAS)} = {len(units) * len(GAMMAS)} 次 MILP；输出 {out}",
          flush=True)

    rows: list[dict] = []
    cache: dict[tuple[str, str, int], dict] = {}
    rng = random.Random(args.brute_seed)
    brute_candidates: list[tuple[str, str, int]] = []
    t0 = time.time()
    for mi, method in enumerate(METHODS):
        for inst in inst_order:
            n, fam, spread = inst_meta(inst)
            for s in SEEDS:
                fp = SCAN_DIR / f"{method}_{inst}_s{s:02d}.json"
                js = json.loads(fp.read_text(encoding="utf-8"))
                eq = np.asarray(js["nominal"]["eq_idx"], int)
                if method == "nominal" and n == 24:
                    brute_candidates.append((method, inst, s))
                for g in GAMMAS:
                    cell = js["cells"][f"d30_g{g}"]
                    S = np.asarray(cell["objsS"], float)[:, eq]
                    ex, gr, rc, ok = exact_mstar(S)
                    rows.append({
                        "instance": inst, "family": fam, "spread": spread, "n_target": n,
                        "method": method, "seed": s, "gamma": g,
                        "eq_size": int(len(eq)), "pool_size": int(len(js["nominal"]["objs0"])),
                        "mstar_exact": ex, "milp_ok": ok,
                        "mstar_greedy": gr, "rc_full": rc, "eps": EPS,
                        "ratio_exact_over_greedy": (ex / gr) if (ex is not None and gr) else None,
                    })
                    cache[(method, inst, s, g)] = {"S": S, "ex": ex, "greedy": gr, "rc": rc}
        print(f"  [{mi + 1}/{len(METHODS)}] {method} 完（{time.time() - t0:.1f}s）", flush=True)

    n1 = write_rows(out / "v3_mstar.csv",
                    ["instance", "family", "spread", "n_target", "method", "seed", "gamma",
                     "eq_size", "pool_size", "mstar_exact", "milp_ok", "mstar_greedy",
                     "rc_full", "eps", "ratio_exact_over_greedy"], rows)
    print(f"v3_mstar.csv 写 {n1} 行", flush=True)

    # ---------- 分布
    dist_rows = []
    for n in (24, 48, 100):
        for method in METHODS:
            for g in GAMMAS:
                rr = [r for r in rows if r["n_target"] == n and r["method"] == method
                      and r["gamma"] == g]
                ex = [r["mstar_exact"] for r in rr if r["mstar_exact"] is not None]
                gr = [r["mstar_greedy"] for r in rr]
                if not rr:
                    continue
                row = {
                    "n_target": n, "method": method, "gamma": g, "n_units": len(rr),
                    "mstar_median": med(ex), "mstar_mean": float(np.mean(ex)) if ex else float("nan"),
                    "mstar_min": min(ex) if ex else None, "mstar_max": max(ex) if ex else None,
                    "share_mstar_1": float(np.mean([v == 1 for v in ex])) if ex else float("nan"),
                    "share_mstar_le2": float(np.mean([v <= 2 for v in ex])) if ex else float("nan"),
                    "share_mstar_le3": float(np.mean([v <= 3 for v in ex])) if ex else float("nan"),
                    "greedy_median": med(gr), "greedy_mean": float(np.mean(gr)) if gr else float("nan"),
                    "greedy_max": max(gr) if gr else None,
                    "ratio_median": med([r["ratio_exact_over_greedy"] for r in rr]),
                    "eq_size_median": med([r["eq_size"] for r in rr]),
                }
                # 直方图：m* 精确值 1..HIST_M*，贪心 1..HIST_G 且 >HIST_G 单列汇总
                for k in range(1, HIST_M + 1):
                    row[f"n_mstar_{k}"] = int(sum(v == k for v in ex))
                row["n_mstar_gt"] = int(sum(v > HIST_M for v in ex))
                for k in range(1, HIST_G + 1):
                    row[f"n_greedy_{k}"] = int(sum(v == k for v in gr))
                row["n_greedy_gt"] = int(sum(v > HIST_G for v in gr))
                dist_rows.append(row)
    hist_fields = [f"n_mstar_{k}" for k in range(1, HIST_M + 1)] + ["n_mstar_gt"]
    hist_fields += [f"n_greedy_{k}" for k in range(1, HIST_G + 1)] + ["n_greedy_gt"]
    fields_dist = ["n_target", "method", "gamma", "n_units", "mstar_median", "mstar_mean",
                   "mstar_min", "mstar_max", "share_mstar_1", "share_mstar_le2",
                   "share_mstar_le3", "greedy_median", "greedy_mean", "greedy_max",
                   "ratio_median", "eq_size_median"] + hist_fields
    n2 = write_rows(out / "v3_mstar_dist.csv", fields_dist, dist_rows)
    print(f"v3_mstar_dist.csv 写 {n2} 行（直方图列到 {hist_fields[-1]}）", flush=True)

    # ---------- 暴力核对（随机 5 个 24 目标池 + 5 个 m*>1 的非平凡池，δ=0.3, Γ=2）
    rng.shuffle(brute_candidates)
    picked = []
    for key in brute_candidates:
        rec = cache[(key[0], key[1], key[2], 2)]
        if rec["ex"] is not None and rec["ex"] <= BRUTE_CAP:
            picked.append(key)
        if len(picked) == BRUTE_N:
            break
    nontrivial = [k for k, rec in cache.items()
                  if k[3] == 2 and rec["ex"] is not None and 2 <= rec["ex"] <= BRUTE_CAP
                  and rec["S"].shape[1] <= 60]
    rng.shuffle(nontrivial)
    picked2 = [(k[0], k[1], k[2]) for k in nontrivial[:BRUTE_N]]
    brute_rows = []
    for selection, keys in (("random_24", picked), ("mstar_ge2", picked2)):
        for method, inst, s in keys:
            rec = cache[(method, inst, s, 2)]
            bm, n_checked = brute_mstar(rec["S"])
            brute_rows.append({
                "selection": selection,
                "instance": inst, "method": method, "seed": s, "gamma": 2,
                "eq_size": int(rec["S"].shape[1]),
                "mstar_milp": rec["ex"], "mstar_brute": bm, "brute_cap": BRUTE_CAP,
                "brute_fully_determined": bm is not None,
                "agree": (bm is not None and bm == rec["ex"]), "n_subsets_checked": n_checked,
            })
            print(f"  [brute:{selection}] {method} {inst} s{s:02d}: MILP={rec['ex']} "
                  f"暴力={bm} 子集={n_checked} 一致={bm == rec['ex']}", flush=True)
    n3 = write_rows(out / "v3_mstar_brute_check.csv",
                    ["selection", "instance", "method", "seed", "gamma", "eq_size",
                     "mstar_milp", "mstar_brute", "brute_cap", "brute_fully_determined",
                     "agree", "n_subsets_checked"], brute_rows)
    print(f"v3_mstar_brute_check.csv 写 {n3} 行", flush=True)

    # ---------- 人读摘要
    lines = ["v3 精确覆盖数 m*（δ=0.3，eps=0.01，名义等价带内；MILP/HiGHS）", "",
             f"数据：{SCAN_DIR.parent}（只读）；{len(rows)} 次 MILP；用时 {time.time() - t_start:.1f}s", ""]
    lines.append("每格：m* 中位 [Q25,Q75] / 贪心中位 / m*=1 比例（n=10 种子×该规模实例数）")
    lines.append("规模  方法          Γ=1              Γ=2              Γ=3              Γ=12")
    for n in (24, 48, 100):
        for method in METHODS:
            cells = []
            ok_all = True
            for g in GAMMAS:
                r = next((x for x in dist_rows if x["n_target"] == n and x["method"] == method
                          and x["gamma"] == g), None)
                if r is None:
                    ok_all = False
                    break
                rr = [x["mstar_exact"] for x in rows if x["n_target"] == n
                      and x["method"] == method and x["gamma"] == g]
                cells.append(f"{r['mstar_median']:.0f} [{q25(rr):.0f},{q75(rr):.0f}]/"
                             f"{r['greedy_median']:.0f}/{r['share_mstar_1']:.2f}")
            if ok_all:
                lines.append(f"{n:>4}  {SHORT[method]:12s} " + " ".join(f"{c:16s}" for c in cells))
    lines.append("")
    lines.append("读法：Γ 越小（扰动越集中）m* 越可能 >1；Γ=12 是单场景全缩水，m*=1 必然。")
    lines.append("贪心停止数远大于 m* 的格 = 贪心菜单严重高估所需方案数。")
    lines.append("")
    bad = [r for r in rows if not r["milp_ok"]]
    lines.append(f"MILP 失败单元：{len(bad)} / {len(rows)}")
    lines.append(f"暴力核对：{sum(r['agree'] for r in brute_rows)}/{len(brute_rows)} 一致"
                 f"（上限 {BRUTE_CAP}，全部完全确定="
                 f"{all(r['brute_fully_determined'] for r in brute_rows)}；"
                 f"随机 24 目标池 5 个 + m*≥2 非平凡池 5 个）")
    (out / "v3_mstar_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:6]), flush=True)
    print(f"全部完成，用时 {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
