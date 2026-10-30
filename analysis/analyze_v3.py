# -*- coding: utf-8 -*-
"""v3 批次统一分析（2026-10-19，阶段0 地基）：只读 20261013_v3 原始数据，产出论文表格/图
脚本的同源 CSV。

输入（只读，绝不修改）：
  data/results/20261013_v3/pools/   1800 npz = 30 实例 × 6 方法 × 10 种子
  data/results/20261013_v3/scans/   1800 json = 同 stem，16 格公共扰动网格
  data/results/20261013_v3/reopt/   1980 npz = 6 实例 × 66 角点 × 5 种子

输出（新目录 data/results/20261013_v3_analysis/）：
  coverage_report.txt / coverage.csv          覆盖率与协议一致性核对
  v3_nominal.csv                              逐单元名义最优/等价带/结构互异数
  v3_degradation.csv                          主工作点 d30_g2 带内平均最坏退化
  v3_rc.csv                                   主工作点 RC5 / 整池 RC / 最优单解 WC
  v3_grid.csv                                 16 格 × 1800 单元长表
  v3_phenomenon.csv (+ _scale.csv)            名义池现象（秩相关/top5 质量差）
  v3_stats.csv                                配对 Wilcoxon + Holm（band/strict vs 其余四法）
  v3_traces.csv                               收敛曲线摘要（50 分位点中位数/IQR）
  v3_summary_method_scale.csv                 六方法 × 三规模中位数（报告/表格直接用）
  v3_reopt.csv / v3_reopt_menu.csv / v3_reopt_summary.txt   角点重解分析

口径约定（与旧脚本对齐，写死在脚本里，改口径必须改这里）：
  - 名义等价带 band_rel = 0.005（core.robust_metrics.equiv_set）。
  - 带内平均最坏退化 = mean_{x∈eq} deg(x)，deg 取扫描 JSON 的 cell["deg"]（= obj0 - min_s obj_s）。
  - RC(X) = min_s max_{x∈X} obj_s(x)（core.robust_metrics.robust_coverage）。
  - RC5 = eq 上 greedy_rc_select(m=5) 纯贪心（无结构破平，与 analyze_v2 历史口径一致）。
  - 整池 RC / 最优单解 WC 直接取扫描 JSON 的 rc_full_pool / best_single_wc。
  - 跨方法鲁棒对比一律用 scans 的公共网格；pools npz 的 objs 语义因方法而异，不跨方法比。

跑法（仓库根）：
  py -3.11 -X utf8 analysis/analyze_v3.py                 # 全量
  py -3.11 -X utf8 analysis/analyze_v3.py --limit 2       # 冒烟（只前 2 个实例）
  py -3.11 -X utf8 analysis/analyze_v3.py --skip-menu     # 跳过角点菜单交叉评估
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr, wilcoxon

HERE = Path(__file__).resolve().parent          # analysis/（analyze_v3 系列平级 import 用）
ROOT = HERE.parent                              # 项目根（core 包用）
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from core.robust_metrics import equiv_set, greedy_rc_select, robust_coverage, scan_cell  # noqa: E402

RES = ROOT / "data" / "results" / "20261013_v3"
POOL_DIR = RES / "pools"
SCAN_DIR = RES / "scans"
REOPT_DIR = RES / "reopt"
DEFAULT_OUT = ROOT / "data" / "results" / "20261013_v3_analysis"

METHODS = ["bcrea_band", "bcrea_strict", "sampled_wc", "nominal", "pessimistic", "nsga2"]
SHORT = {
    "bcrea_band": "BCREA-band",
    "bcrea_strict": "BCREA-strict",
    "sampled_wc": "sampled-WC",
    "nominal": "nominal",
    "pessimistic": "pessimistic",
    "nsga2": "NSGA-II",
}
# 文件名方法标签 → 池 config 里的 method 标签
METHOD_LABEL = {
    "nominal": "nominal_kcea",
    "pessimistic": "pessimistic_resolve",
    "nsga2": "nsga2",
    "sampled_wc": "sampled_worst_case",
    "bcrea_band": "bcrea_band",
    "bcrea_strict": "bcrea_strict",
}
SCEN_SEED = {"bcrea_band": 9001, "bcrea_strict": 9001, "nsga2": 9001, "sampled_wc": 9101}
DELTA_GAMMA = {"bcrea_band": (0.3, 2), "bcrea_strict": (0.3, 2), "nsga2": (0.3, 2),
               "sampled_wc": (0.3, 2), "pessimistic": (0.3, 12)}

SEEDS = list(range(1, 11))
BAND_REL = 0.005
MAIN_CELL = "d30_g2"
DELTAS = [0.1, 0.2, 0.3, 0.4]
GAMMAS = [1, 2, 3, 12]
CELL_KEYS = [f"d{int(round(d * 100)):02d}_g{g}" for d in DELTAS for g in GAMMAS]
N_SCEN = {1: 12, 2: 66, 3: 220, 12: 1}
SCAN_SCEN_SEED = 6000
REOPT_PAIRS = list(combinations(range(12), 2))
N_COMP = 12
N_POINT = 50          # 收敛曲线分位点数
M_RC = 5

RE_REOPT = re.compile(r"^reopt_c(\d+)c(\d+)_(.+)_s(\d+)$")

# 与 make_tasks_v3.py:45-47 的 REOPT_INSTANCES 一致：三规模 × 两种散布各取 01 号
REOPT_INSTANCES = [f"gen{n}_{m}_01" for n in (24, 48, 100) for m in ("conc", "disp")]


# ---------------------------------------------------------------- 基础工具
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


def resample(trace: np.ndarray, n: int = N_POINT) -> np.ndarray:
    t = np.asarray(trace, float).ravel()
    if t.size == 0:
        return np.full(n, np.nan)
    if t.size == 1:
        return np.full(n, float(t[0]))
    return np.interp(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, t.size), t)


def parse_stem(stem: str):
    for m in METHODS:
        pre = m + "_"
        if stem.startswith(pre):
            rest = stem[len(pre):]
            if "_s" not in rest:
                return None
            inst, seed = rest.rsplit("_s", 1)
            try:
                return m, inst, int(seed)
            except ValueError:
                return None
    return None


def tevc_k(inst: str) -> int:
    return int(inst[1:])


def inst_meta(inst: str) -> tuple[int, str, str]:
    """→ (目标数, 族 gen/tevc, 散布 conc/disp/na)。"""
    if inst.startswith("gen"):
        head, mode, _rep = inst.split("_")
        return int(head[3:]), "gen", mode
    n = 100 if inst == "i100" else 48 if inst == "i11" else 24
    return n, "tevc", "na"


def scenario_path(inst: str) -> str:
    n, fam, _ = inst_meta(inst)
    if fam == "gen":
        return f"data/scenarios/generated/{inst}.mat"
    return f"data/scenarios/tevc/Info_enemy_{tevc_k(inst)}.mat"


def build_instance_order() -> list[str]:
    """按规模 → gen(conc/disp 01-03) → tevc 的顺序排，便于人读。"""
    order: list[str] = []
    tevc = [f"i{k}" for k in range(1, 11)] + ["i11", "i100"]
    for n in (24, 48, 100):
        for mode in ("conc", "disp"):
            order.extend(f"gen{n}_{mode}_{r:02d}" for r in range(1, 4))
        order.extend(sorted([t for t in tevc if inst_meta(t)[0] == n], key=tevc_k))
    return order


# ---------------------------------------------------------------- 一致性核对
CHECKS = ["parse_ok", "label_ok", "seed_ok", "n_target_ok", "budget_ok", "actual_ok",
          "scenario_ok", "delta_gamma_ok", "band_ok", "scen_seed_ok", "pool_size_ok",
          "provenance_ok", "nominal_match_ok", "eq_ok", "cells_ok", "struct_ok"]
HARD = {"parse_ok", "label_ok", "seed_ok", "n_target_ok", "budget_ok", "actual_ok",
        "scenario_ok", "delta_gamma_ok", "band_ok", "scen_seed_ok", "pool_size_ok",
        "provenance_ok", "cells_ok"}


def new_check_row(kind: str, fp: Path, method, inst, seed) -> dict:
    row = {"kind": kind, "file": fp.name, "instance": inst, "method": method, "seed": seed,
           "status": "", "note": ""}
    for c in CHECKS:
        row[c] = None
    return row


def finish_row(row: dict, notes: list[str]) -> dict:
    hard_fail = [c for c in HARD if row.get(c) is not None and not row[c]]
    soft_fail = [c for c in CHECKS if c not in HARD and row.get(c) is not None and not row[c]]
    if hard_fail:
        row["status"] = "FAIL"
        notes = [f"{c} 失败" for c in hard_fail] + notes
    elif soft_fail:
        row["status"] = "WARN"
        notes = [f"{c} 不一致" for c in soft_fail] + notes
    else:
        row["status"] = "PASS"
    row["note"] = "；".join(notes)
    return row


def load_pool(fp: Path):
    z = np.load(fp, allow_pickle=False)
    keys = set(z.files)
    cfg = json.loads(str(z["config"]))
    return z, keys, cfg


def check_pool(fp: Path, method: str, inst: str, seed: int) -> tuple[dict | None, dict]:
    row = new_check_row("pool", fp, method, inst, seed)
    n_exp, _fam, _sp = inst_meta(inst)
    if not fp.exists():
        row["parse_ok"] = False
        return None, finish_row(row, ["文件缺失"])
    row["parse_ok"] = True
    notes: list[str] = []
    z, keys, cfg = load_pool(fp)
    decs = np.asarray(z["decs"])
    objs = np.asarray(z["objs"], float)
    row["label_ok"] = cfg.get("method") == METHOD_LABEL[method]
    if not row["label_ok"]:
        notes.append(f"method={cfg.get('method')}")
    row["seed_ok"] = cfg.get("seed") == seed
    row["n_target_ok"] = cfg.get("n_target") == n_exp and decs.shape[2] == n_exp
    row["budget_ok"] = cfg.get("decode_budget") == 9000 * n_exp
    act = cfg.get("actual_decode_evals")
    tot = cfg.get("total_evals")
    if method == "sampled_wc":
        gap = (cfg.get("decode_budget") or 0) - (act or 0)
        row["actual_ok"] = act is not None and 0 <= gap <= 17 * (cfg.get("pop_size") or 0) + 1
    else:
        row["actual_ok"] = act is not None and act == cfg.get("decode_budget") and tot == act
    row["scenario_ok"] = str(cfg.get("scenario", "")).replace("\\", "/") == scenario_path(inst)
    dg = DELTA_GAMMA.get(method)
    if dg is None:
        row["delta_gamma_ok"] = None
    else:
        row["delta_gamma_ok"] = (abs(float(cfg.get("delta", -1)) - dg[0]) < 1e-12
                                 and int(cfg.get("gamma", -1)) == dg[1])
    if "eps_band_rel" in cfg:
        row["band_ok"] = abs(float(cfg["eps_band_rel"]) - BAND_REL) < 1e-12
    else:
        row["band_ok"] = None
    if method in SCEN_SEED:
        row["scen_seed_ok"] = cfg.get("scen_seed") == SCEN_SEED[method]
    else:
        row["scen_seed_ok"] = None
    row["pool_size_ok"] = (decs.shape[0] == objs.shape[0] > 0
                           and decs.shape[0] == (cfg.get("front1_size", cfg.get("pop_size"))
                                                 if method == "nsga2" else cfg.get("pop_size")))
    # 结构互异数在 scan 侧用扫描的 eq + 池 decs 复算（pessimistic 的池 objs 是悲观值，不能在这里算）
    rec = {
        "cfg": cfg,
        "best": float(objs.max()) if len(objs) else float("nan"),
        "pool_size": int(decs.shape[0]),
        "actual_decode_evals": int(act) if act is not None else -1,
        "budget": int(cfg.get("decode_budget") or 0),
        "series": {},
    }
    tr = np.asarray(z["trace"], float) if "trace" in keys else None
    if tr is not None:
        if method == "pessimistic":
            rec["series"]["pessimistic_obj"] = resample(tr)
        elif method == "sampled_wc":
            rec["series"]["sampled_wc_fit"] = resample(tr)
        else:
            rec["series"]["nominal_obj"] = resample(tr)
        if "trace_deg" in keys:
            rec["series"]["in_band_deg"] = resample(np.asarray(z["trace_deg"], float))
    return rec, finish_row(row, notes)


def check_scan(fp: Path, method: str, inst: str, seed: int, pool: dict | None) -> tuple[dict | None, dict]:
    row = new_check_row("scan", fp, method, inst, seed)
    n_exp, _fam, _sp = inst_meta(inst)
    if not fp.exists():
        row["parse_ok"] = False
        return None, finish_row(row, ["文件缺失"])
    row["parse_ok"] = True
    notes: list[str] = []
    js = json.loads(fp.read_text(encoding="utf-8"))
    cfg = js.get("config") or {}
    row["label_ok"] = None                       # 扫描 JSON 无 method 字段，用 provenance 校验
    row["seed_ok"] = None
    row["n_target_ok"] = cfg.get("n_target") == n_exp
    row["budget_ok"] = None
    row["actual_ok"] = None
    row["scenario_ok"] = str(cfg.get("scenario", "")).replace("\\", "/") == scenario_path(inst)
    row["delta_gamma_ok"] = (cfg.get("deltas") == DELTAS and cfg.get("gammas") == GAMMAS
                             and cfg.get("scen_seed") == SCAN_SCEN_SEED)
    row["band_ok"] = abs(float(cfg.get("band_rel", -1)) - BAND_REL) < 1e-12
    row["scen_seed_ok"] = cfg.get("scen_seed") == SCAN_SCEN_SEED
    row["pool_size_ok"] = None
    prov = js.get("pool_provenance") or {}
    prov_cfg = prov.get("config") or {}
    exp_pool = f"data/results/20261013_v3/pools/{fp.stem}.npz"
    prov_file = str(prov.get("pool_file", "")).replace("\\", "/").lstrip("./")
    prov_ok = prov_file == exp_pool.lstrip("./")
    if pool is not None:
        pc = pool["cfg"]
        prov_ok = prov_ok and prov_cfg.get("method") == pc.get("method") \
            and prov_cfg.get("seed") == pc.get("seed") \
            and prov_cfg.get("decode_budget") == pc.get("decode_budget") \
            and prov_cfg.get("actual_decode_evals") == pc.get("actual_decode_evals")
    row["provenance_ok"] = prov_ok
    if not prov_ok:
        notes.append(f"provenance={prov_file}")
    nom = js.get("nominal") or {}
    objs0 = np.asarray(nom.get("objs0", []), float)
    eq = np.asarray(nom.get("eq_idx", []), int)
    best = float(nom.get("best", float("nan")))
    tol = 1e-6 * max(1.0, abs(best)) if np.isfinite(best) else 1.0
    ok_best = np.isfinite(best) and len(objs0) > 0 and abs(best - float(objs0.max())) <= tol
    if pool is not None and method != "pessimistic":
        ok_best = ok_best and abs(best - pool["best"]) <= tol
    row["nominal_match_ok"] = bool(ok_best)
    ok_eq = len(eq) > 0 and len(objs0) > 0 and np.array_equal(eq, equiv_set(objs0, BAND_REL))
    if not ok_eq and len(eq) and len(objs0):
        ok_eq = set(eq.tolist()) == set(equiv_set(objs0, BAND_REL).tolist())
    if not ok_eq and len(eq) and len(objs0):
        # JSON 的 objs0 只有 6 位小数，恰好压在带边界上的成员可能差 1 个；判定为可接受的舍入差
        e_re = set(equiv_set(objs0, BAND_REL).tolist())
        diff = set(eq.tolist()) ^ e_re
        thr = float(objs0.max()) * (1.0 - BAND_REL)
        if len(diff) <= 2 and all(abs(float(objs0[i]) - thr) <= 1e-6 for i in diff):
            ok_eq = True
            notes.append(f"等价带边界舍入差 {len(diff)} 个")
    row["eq_ok"] = bool(ok_eq)
    cells = js.get("cells") or {}
    ok_cells = set(cells) == set(CELL_KEYS)
    if ok_cells:
        for ck in CELL_KEYS:
            c = cells[ck]
            if (abs(float(c.get("delta", -1)) - float(ck[1:3]) / 100.0) > 1e-12
                    or int(c.get("gamma", -1)) != int(ck.split("_g")[1])):
                ok_cells = False
                break
            ns = int(c.get("n_scen_used", -1))
            if ns != N_SCEN[int(ck.split("_g")[1])]:
                ok_cells = False
                break
            S = c.get("objsS") or []
            if len(S) != ns or (ns and len(S[0]) != len(objs0)):
                ok_cells = False
                break
    row["cells_ok"] = bool(ok_cells)
    if not ok_cells:
        notes.append("16 格结构异常")
    if pool is not None:
        # 结构互异数：用扫描的 eq（名义口径）索引池 decs 复算；pessimistic 的池 objs 是悲观值也能用
        d = np.load(POOL_DIR / f"{method}_{inst}_s{seed:02d}.npz", allow_pickle=False)
        decs = np.asarray(d["decs"])
        flat = decs[eq].reshape(len(eq), -1)
        n_struct = int(np.unique(flat, axis=0).shape[0])
        row["struct_ok"] = (int(nom.get("n_struct_distinct_eq", -1)) == n_struct)
        if not row["struct_ok"]:
            notes.append(f"结构互异数 scan={nom.get('n_struct_distinct_eq')} 复算={n_struct}")
    return finish_row(row, notes)


def check_reopt(fp: Path, a: int, b: int, inst: str, seed: int) -> tuple[dict | None, dict]:
    row = new_check_row("reopt", fp, "true_reoptimization", inst, seed)
    n_exp, _fam, _sp = inst_meta(inst)
    if not fp.exists():
        row["parse_ok"] = False
        return None, finish_row(row, ["文件缺失"])
    row["parse_ok"] = True
    notes: list[str] = []
    z = np.load(fp, allow_pickle=False)
    cfg = json.loads(str(z["config"]))
    objs = np.asarray(z["objs"], float)
    row["label_ok"] = cfg.get("method") == "true_reoptimization"
    row["seed_ok"] = cfg.get("seed") == seed
    row["n_target_ok"] = cfg.get("n_target") == n_exp and np.asarray(z["decs"]).shape[2] == n_exp
    row["budget_ok"] = cfg.get("decode_budget") == 9000 * n_exp
    row["actual_ok"] = cfg.get("actual_decode_evals") == cfg.get("decode_budget")
    row["scenario_ok"] = str(cfg.get("scenario", "")).replace("\\", "/") == scenario_path(inst)
    row["delta_gamma_ok"] = (abs(float(cfg.get("delta", -1)) - 0.3) < 1e-12
                             and int(cfg.get("gamma", -1)) == 2
                             and list(cfg.get("components") or []) == [a, b])
    row["band_ok"] = None
    row["scen_seed_ok"] = None
    row["pool_size_ok"] = len(objs) > 0 and np.asarray(z["decs"]).shape[0] == len(objs) \
        and len(objs) == cfg.get("pop_size")
    row["provenance_ok"] = None
    rec = {
        "best": float(objs.max()) if len(objs) else float("nan"),
        "elapsed_s": float(cfg.get("elapsed_s", float("nan"))),
        "decs": np.asarray(z["decs"], np.int16),
        "objs": objs,
    }
    return rec, finish_row(row, notes)


# ---------------------------------------------------------------- 统计
def holm_adjust(pvals: list[float]) -> np.ndarray:
    p = np.asarray(pvals, float)
    m = len(p)
    adj = np.full(m, np.nan)
    idx = [i for i in range(m) if np.isfinite(p[i])]
    if not idx:
        return adj
    order = sorted(idx, key=lambda i: p[i])
    prev = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (len(order) - rank) * p[i])
        val = max(val, prev)
        adj[i] = val
        prev = val
    return adj


def paired_test(a: np.ndarray, b: np.ndarray, higher_better: bool) -> dict:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    n = len(d)
    nz = int(np.count_nonzero(d))
    n_zero = n - nz
    ad = np.abs(d[d != 0])
    n_tied = int(len(ad) - len(np.unique(np.round(ad, 12))))
    if higher_better:
        wins = int(np.sum(d > 0))
        losses = int(np.sum(d < 0))
    else:
        wins = int(np.sum(d < 0))
        losses = int(np.sum(d > 0))
    res = {"n_pairs": n, "n_nonzero": nz, "n_zero": n_zero, "n_abs_tied": n_tied,
           "median_variant": med(a), "median_opponent": med(b), "median_diff": med(d),
           "wins": wins, "losses": losses, "ties": n - wins - losses,
           "wilcoxon_stat": float("nan"), "p_value": float("nan"), "rank_biserial": float("nan")}
    if nz >= 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                r = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
                res["wilcoxon_stat"] = float(r.statistic)
                res["p_value"] = float(r.pvalue)
            except Exception:
                pass
        ranks = rankdata(ad)
        res["rank_biserial"] = float((ranks[d[d != 0] > 0].sum()
                                      - ranks[d[d != 0] < 0].sum()) / ranks.sum())
    return res


# ---------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description="v3 批次统一分析")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=0, help="冒烟：只取实例顺序前 N 个")
    ap.add_argument("--skip-menu", action="store_true", help="跳过角点专才菜单交叉评估")
    ap.add_argument("--force-menu", action="store_true", help="强制重算菜单交叉评估（忽略缓存）")
    ap.add_argument("--backend", choices=["auto", "cpu", "gpu"], default="auto")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    inst_order = build_instance_order()
    if args.limit:
        inst_order = inst_order[: args.limit]
    units = [(inst, s) for inst in inst_order for s in SEEDS]
    print(f"实例 {len(inst_order)} 个 × 种子 {len(SEEDS)} = {len(units)} 单元；输出到 {out}", flush=True)

    # ================= A. 覆盖率与一致性核对（pools → scans → reopt）
    cov_rows: list[dict] = []
    pool_rec: dict[tuple[str, str, int], dict] = {}
    t0 = time.time()
    for mi, method in enumerate(METHODS):
        for inst in inst_order:
            for s in SEEDS:
                fp = POOL_DIR / f"{method}_{inst}_s{s:02d}.npz"
                rec, row = check_pool(fp, method, inst, s)
                cov_rows.append(row)
                if rec is not None:
                    pool_rec[(method, inst, s)] = rec
        print(f"  [pools {mi+1}/{len(METHODS)}] {method} 完（{time.time()-t0:.1f}s）", flush=True)

    scan_ok: set[tuple[str, str, int]] = set()
    t0 = time.time()
    for mi, method in enumerate(METHODS):
        for inst in inst_order:
            for s in SEEDS:
                fp = SCAN_DIR / f"{method}_{inst}_s{s:02d}.json"
                row = check_scan(fp, method, inst, s, pool_rec.get((method, inst, s)))
                cov_rows.append(row)
                if row["parse_ok"]:
                    scan_ok.add((method, inst, s))
        print(f"  [scans {mi+1}/{len(METHODS)}] {method} 完（{time.time()-t0:.1f}s）", flush=True)

    cov_rows.sort(key=lambda r: (r["kind"], r["instance"], r["method"], r["seed"]))
    fields_cov = ["kind", "file", "instance", "method", "seed"] + CHECKS + ["status", "note"]
    n_cov = write_rows(out / "coverage.csv", fields_cov, cov_rows)
    print(f"coverage.csv 写 {n_cov} 行", flush=True)

    # ================= B. 核心指标
    rows_nom, rows_deg, rows_rc, rows_grid, rows_phen = [], [], [], [], []
    per_unit: dict[tuple[str, str, int], dict] = {}
    missing_scans: list[str] = []

    for method in METHODS:
        for inst in inst_order:
            n, fam, spread = inst_meta(inst)
            for s in SEEDS:
                key = (method, inst, s)
                if key not in scan_ok:
                    missing_scans.append(f"{method}_{inst}_s{s:02d}")
                    continue
                js = json.loads((SCAN_DIR / f"{method}_{inst}_s{s:02d}.json")
                                .read_text(encoding="utf-8"))
                nom = js["nominal"]
                cells = js["cells"]
                objs0 = np.asarray(nom["objs0"], float)
                eq = np.asarray(nom["eq_idx"], int)
                best = float(nom["best"])
                main = cells[MAIN_CELL]
                deg = np.asarray(main["deg"], float)
                S = np.asarray(main["objsS"], float)
                S_eq = S[:, eq]
                sel = greedy_rc_select(S_eq, M_RC)
                members = eq[np.asarray(sel, int)]
                rc5 = robust_coverage(S, members)
                rc_full = float(main["rc_full_pool"])
                best_single = float(main["best_single_wc"])
                pr = pool_rec.get(key) or {}

                rows_nom.append({
                    "instance": inst, "family": fam, "spread": spread, "n_target": n,
                    "method": method, "seed": s,
                    "pool_size": int(len(objs0)),
                    "best_nominal": best,
                    "eq_size": int(len(eq)),
                    "eq_frac": len(eq) / len(objs0) if len(objs0) else float("nan"),
                    "n_struct_distinct_eq": int(nom["n_struct_distinct_eq"]),
                    "spread_obj0_eq": float(nom["spread_obj0_eq"]),
                    "actual_decode_evals": pr.get("actual_decode_evals", -1),
                })
                rows_deg.append({
                    "instance": inst, "family": fam, "spread": spread, "n_target": n,
                    "method": method, "seed": s,
                    "eq_size": int(len(eq)), "best_nominal": best,
                    "deg_in_band_mean": float(deg[eq].mean()),
                    "deg_pool_mean": float(deg.mean()),
                    "deg_min_pool": float(deg.min()),
                    "deg_min_in_band": float(deg[eq].min()),
                })
                rows_rc.append({
                    "instance": inst, "family": fam, "spread": spread, "n_target": n,
                    "method": method, "seed": s,
                    "rc5": rc5, "rc5_m": int(len(members)),
                    "rc5_over_full": rc5 / rc_full if rc_full else float("nan"),
                    "rc5_gain_over_single": rc5 - best_single,
                    "rc_full_pool": rc_full, "best_single_wc": best_single,
                    "nominal_best": best,
                })
                for ck in CELL_KEYS:
                    c = cells[ck]
                    cdeg = np.asarray(c["deg"], float)
                    rows_grid.append({
                        "cell": ck, "delta": float(c["delta"]), "gamma": int(c["gamma"]),
                        "instance": inst, "family": fam, "spread": spread, "n_target": n,
                        "method": method, "seed": s,
                        "n_scen_used": int(c["n_scen_used"]),
                        "deg_in_band_mean": float(cdeg[eq].mean()),
                        "rc_full_pool": float(c["rc_full_pool"]),
                        "best_single_wc": float(c["best_single_wc"]),
                        "rel_spread": float(c["rel_spread"]),
                        "reorders": int(c["reorders"]),
                        "n_distinct_worst": int(c["n_distinct_worst"]),
                        "spread_wc": float(c["spread_wc"]),
                        "amp": float(c["amp"]) if c.get("amp") is not None else float("nan"),
                    })

                per_unit[key] = {
                    "rc5": rc5, "deg_in_band_mean": float(deg[eq].mean()),
                    "best_nominal": best, "rc_full_pool": rc_full,
                    "best_single_wc": best_single, "eq_size": int(len(eq)),
                }

                # 现象（名义池性质）：只用 nominal 方法
                if method == "nominal":
                    o0 = objs0[eq]
                    wc = objs0[eq] - deg[eq]
                    tied = float(o0.max() - o0.min()) < 1e-9 if len(eq) else True
                    rho = float("nan")
                    if not tied and len(eq) > 2:
                        rho = float(spearmanr(o0, wc).statistic)
                    k = min(5, len(eq))
                    top_n = np.argsort(-o0, kind="stable")[:k]
                    top_r = np.argsort(-wc, kind="stable")[:k]
                    jac = len(set(top_n.tolist()) & set(top_r.tolist())) / len(set(top_n.tolist()) | set(top_r.tolist()))
                    m_nom = eq[top_n]
                    qual_n = float(S[:, m_nom].max(axis=1).mean()) if k else float("nan")
                    qual_r = float(S[:, members].max(axis=1).mean()) if len(members) else float("nan")
                    rc_n = robust_coverage(S, m_nom) if k else float("nan")
                    rows_phen.append({
                        "instance": inst, "family": fam, "spread": spread, "n_target": n,
                        "seed": s, "n_eq": int(len(eq)),
                        "eq_frac": len(eq) / len(objs0) if len(objs0) else float("nan"),
                        "nominal_all_tied": bool(tied), "spearman_nom_wc": rho,
                        "jaccard_top5": jac,
                        "mean_qual_nomtop5": qual_n, "mean_qual_rc5": qual_r,
                        "mean_qual_gap": qual_r - qual_n,
                        "rc_nomtop5": rc_n, "rc_rc5": rc5, "rc_gap": rc5 - rc_n,
                        "rel_spread": float(main["rel_spread"]),
                    })
        print(f"  [metrics] {method} 完", flush=True)

    n1 = write_rows(out / "v3_nominal.csv",
                    ["instance", "family", "spread", "n_target", "method", "seed", "pool_size",
                     "best_nominal", "eq_size", "eq_frac", "n_struct_distinct_eq",
                     "spread_obj0_eq", "actual_decode_evals"], rows_nom)
    n2 = write_rows(out / "v3_degradation.csv",
                    ["instance", "family", "spread", "n_target", "method", "seed", "eq_size",
                     "best_nominal", "deg_in_band_mean", "deg_pool_mean", "deg_min_pool",
                     "deg_min_in_band"], rows_deg)
    n3 = write_rows(out / "v3_rc.csv",
                    ["instance", "family", "spread", "n_target", "method", "seed", "rc5", "rc5_m",
                     "rc5_over_full", "rc5_gain_over_single", "rc_full_pool", "best_single_wc",
                     "nominal_best"], rows_rc)
    n4 = write_rows(out / "v3_grid.csv",
                    ["cell", "delta", "gamma", "instance", "family", "spread", "n_target",
                     "method", "seed", "n_scen_used", "deg_in_band_mean", "rc_full_pool",
                     "best_single_wc", "rel_spread", "reorders", "n_distinct_worst",
                     "spread_wc", "amp"], rows_grid)
    n5 = write_rows(out / "v3_phenomenon.csv",
                    ["instance", "family", "spread", "n_target", "seed", "n_eq", "eq_frac",
                     "nominal_all_tied", "spearman_nom_wc", "jaccard_top5",
                     "mean_qual_nomtop5", "mean_qual_rc5", "mean_qual_gap", "rc_nomtop5",
                     "rc_rc5", "rc_gap", "rel_spread"], rows_phen)
    print(f"写 CSV：nominal {n1} / degradation {n2} / rc {n3} / grid {n4} / phenomenon {n5}",
          flush=True)

    # ---- 现象按规模聚合
    phen_scale = []
    for n in (24, 48, 100):
        rr = [r for r in rows_phen if r["n_target"] == n]
        if not rr:
            continue
        rhos = [r["spearman_nom_wc"] for r in rr]
        phen_scale.append({
            "n_target": n, "n_units": len(rr),
            "share_nominal_all_tied": float(np.mean([r["nominal_all_tied"] for r in rr])),
            "spearman_median": med(rhos),
            "spearman_mean": float(np.nanmean(rhos)) if any(np.isfinite(x) for x in rhos) else float("nan"),
            "jaccard_top5_median": med([r["jaccard_top5"] for r in rr]),
            "mean_qual_gap_median": med([r["mean_qual_gap"] for r in rr]),
            "share_rc_gap_positive": float(np.mean([r["rc_gap"] > 0 for r in rr])),
            "rc_gap_median": med([r["rc_gap"] for r in rr]),
            "rel_spread_median": med([r["rel_spread"] for r in rr]),
            "eq_frac_median": med([r["eq_frac"] for r in rr]),
        })
    write_rows(out / "v3_phenomenon_scale.csv",
               ["n_target", "n_units", "share_nominal_all_tied", "spearman_median", "spearman_mean",
                "jaccard_top5_median", "mean_qual_gap_median", "share_rc_gap_positive",
                "rc_gap_median", "rel_spread_median", "eq_frac_median"], phen_scale)

    # ================= 六方法 × 规模中位数（报告/表格直接用）
    rows_sum = []
    for n in (24, 48, 100):
        for method in METHODS:
            sel = [per_unit[(method, i, s)] for (i, s) in units
                   if inst_meta(i)[0] == n and (method, i, s) in per_unit]
            if not sel:
                continue
            rows_sum.append({
                "n_target": n, "method": method, "n_units": len(sel),
                "best_nominal_median": med([r["best_nominal"] for r in sel]),
                "eq_size_median": med([r["eq_size"] for r in sel]),
                "deg_in_band_median": med([r["deg_in_band_mean"] for r in sel]),
                "deg_in_band_mean": float(np.mean([r["deg_in_band_mean"] for r in sel])),
                "rc5_median": med([r["rc5"] for r in sel]),
                "rc_full_median": med([r["rc_full_pool"] for r in sel]),
                "best_single_wc_median": med([r["best_single_wc"] for r in sel]),
            })
    write_rows(out / "v3_summary_method_scale.csv",
               ["n_target", "method", "n_units", "best_nominal_median", "eq_size_median",
                "deg_in_band_median", "deg_in_band_mean", "rc5_median", "rc_full_median",
                "best_single_wc_median"], rows_sum)

    # ================= 配对统计（band/strict vs 其余四法）
    stats_rows = []
    variants = ["bcrea_band", "bcrea_strict"]
    opponents = ["nominal", "pessimistic", "sampled_wc", "nsga2"]
    metrics = [("rc5", "higher"), ("deg_in_band_mean", "lower"), ("best_nominal", "higher")]
    unit_pairs = [(i, s) for (i, s) in units]
    subsets = [("all", lambda i: True), ("24", lambda i: inst_meta(i)[0] == 24),
               ("48", lambda i: inst_meta(i)[0] == 48), ("100", lambda i: inst_meta(i)[0] == 100)]

    for sub_name, pred in subsets:
        sub_units = [(i, s) for (i, s) in unit_pairs if pred(i)]
        for metric, direction in metrics:
            for variant in variants:
                fam_rows = []
                for opp in opponents:
                    a = np.array([per_unit[(variant, i, s)][metric] for (i, s) in sub_units])
                    b = np.array([per_unit[(opp, i, s)][metric] for (i, s) in sub_units])
                    r = paired_test(a, b, higher_better=(direction == "higher"))
                    fam_rows.append({"subset": sub_name, "metric": metric, "direction": direction,
                                     "variant": variant, "opponent": opp, "family": f"{sub_name}|{metric}|{variant}",
                                     **r})
                p_adj = holm_adjust([r["p_value"] for r in fam_rows])
                for r, pa in zip(fam_rows, p_adj):
                    r["p_holm"] = float(pa)
                    r["significant_holm"] = bool(np.isfinite(pa) and pa < 0.05)
                    stats_rows.append(r)
    write_rows(out / "v3_stats.csv",
               ["subset", "family", "metric", "direction", "variant", "opponent", "n_pairs",
                "n_nonzero", "n_zero", "n_abs_tied", "median_variant", "median_opponent",
                "median_diff", "wins", "losses", "ties", "wilcoxon_stat", "p_value", "p_holm",
                "rank_biserial", "significant_holm"], stats_rows)
    print(f"v3_stats.csv 写 {len(stats_rows)} 行", flush=True)

    # ================= 收敛曲线摘要
    trace_rows = []
    for method in METHODS:
        series_names = set()
        for inst in inst_order:
            for s in SEEDS:
                rec = pool_rec.get((method, inst, s))
                if rec:
                    series_names.update(rec["series"].keys())
        for series in sorted(series_names):
            for n in (24, 48, 100):
                mats = []
                budget = 9000 * n
                for inst in inst_order:
                    if inst_meta(inst)[0] != n:
                        continue
                    for s in SEEDS:
                        rec = pool_rec.get((method, inst, s))
                        if rec and series in rec["series"]:
                            mats.append(rec["series"][series])
                if not mats:
                    continue
                A = np.vstack(mats)
                for k in range(N_POINT):
                    frac = k / (N_POINT - 1)
                    col = A[:, k]
                    trace_rows.append({
                        "method": method, "series": series, "n_target": n, "point": k,
                        "budget_frac": frac, "decode_evals": frac * budget, "n_runs": len(col),
                        "median": med(col), "q25": q25(col), "q75": q75(col),
                        "iqr": q75(col) - q25(col),
                    })
    write_rows(out / "v3_traces.csv",
               ["method", "series", "n_target", "point", "budget_frac", "decode_evals",
                "n_runs", "median", "q25", "q75", "iqr"], trace_rows)
    print(f"v3_traces.csv 写 {len(trace_rows)} 行", flush=True)

    # ================= C. reopt
    reopt_rec: dict[tuple[str, int, int, int], dict] = {}
    reopt_rows = []
    for inst in [i for i in REOPT_INSTANCES if i in inst_order]:
        n, fam, spread = inst_meta(inst)
        for a, b in REOPT_PAIRS:
            for s in range(1, 6):
                fp = REOPT_DIR / f"reopt_c{a}c{b}_{inst}_s{s:02d}.npz"
                rec, row = check_reopt(fp, a, b, inst, s)
                cov_rows.append(row)
                if rec is None:
                    continue
                reopt_rec[(inst, s, a, b)] = rec
    # reopt 覆盖行补写（pools/scans 已写，追加到 coverage.csv 尾部前重排）
    cov_rows.sort(key=lambda r: (r["kind"], r["instance"], r["method"], r["seed"],
                                 str(r.get("file"))))
    write_rows(out / "coverage.csv", fields_cov, cov_rows)

    for inst in [i for i in REOPT_INSTANCES if i in inst_order]:
        n, fam, spread = inst_meta(inst)
        for s in range(1, 6):
            nom_pool = pool_rec.get(("nominal", inst, s))
            best_nom = nom_pool["best"] if nom_pool else float("nan")
            for a, b in REOPT_PAIRS:
                rec = reopt_rec.get((inst, s, a, b))
                if rec is None:
                    continue
                gap = rec["best"] - best_nom
                reopt_rows.append({
                    "instance": inst, "family": fam, "spread": spread, "n_target": n,
                    "seed": s, "comp_a": a, "comp_b": b, "comp_pair": f"c{a}c{b}",
                    "best_corner": rec["best"], "best_nominal_pool": best_nom,
                    "gap_corner_minus_nominal": gap,
                    "nominal_cost_rel": (best_nom - rec["best"]) / best_nom if best_nom else float("nan"),
                    "elapsed_s": rec["elapsed_s"], "gamma": 2, "delta": 0.3,
                })
    n6 = write_rows(out / "v3_reopt.csv",
                    ["instance", "family", "spread", "n_target", "seed", "comp_a", "comp_b",
                     "comp_pair", "best_corner", "best_nominal_pool",
                     "gap_corner_minus_nominal", "nominal_cost_rel", "elapsed_s", "gamma",
                     "delta"], reopt_rows)
    print(f"v3_reopt.csv 写 {n6} 行", flush=True)

    # ---- 角点专才菜单交叉评估（重评估已存档解，不跑新搜索）
    menu_rows = []
    cache_fp = out / "reopt_menu_rc.json"
    cache = {}
    if cache_fp.exists() and not args.force_menu:
        cache = json.loads(cache_fp.read_text(encoding="utf-8"))
    if not args.skip_menu:
        import core.robust_metrics as r6
        from core.model import DefenseParams
        from core.scenarios import load_enemy

        backend = r6.resolve_backend(args.backend)
        print(f"角点菜单交叉评估后端={backend}", flush=True)
        for inst in [i for i in REOPT_INSTANCES if i in inst_order]:
            for s in range(1, 6):
                ck = f"{inst}|{s}"
                if ck in cache:
                    continue
                specs, lbs = [], []
                for a, b in REOPT_PAIRS:
                    rec = reopt_rec.get((inst, s, a, b))
                    if rec is None:
                        break
                    i_best = int(np.argmax(rec["objs"]))
                    specs.append(rec["decs"][i_best])
                    lbs.append(float(rec["objs"][i_best]))
                if len(specs) != len(REOPT_PAIRS):
                    continue
                decs_menu = np.stack(specs).astype(np.int16)
                enemy = load_enemy(str(ROOT / scenario_path(inst)))
                _masks, S, _dt = scan_cell(enemy, DefenseParams(), decs_menu, 0.3, 2,
                                           60, SCAN_SCEN_SEED, 250, backend=backend)
                cache[ck] = {
                    "rc_menu": robust_coverage(S, np.arange(len(specs))),
                    "rc_menu_lb": float(min(lbs)),
                    "n_unique_menu": int(len(np.unique(decs_menu.reshape(len(specs), -1), axis=0))),
                }
                print(f"  [menu] {ck} rc={cache[ck]['rc_menu']:.4f} lb={cache[ck]['rc_menu_lb']:.4f}",
                      flush=True)
        cache_fp.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    for inst in [i for i in REOPT_INSTANCES if i in inst_order]:
        n, fam, spread = inst_meta(inst)
        for s in range(1, 6):
            ck = f"{inst}|{s}"
            rec = cache.get(ck)
            rc_n = (per_unit.get(("nominal", inst, s)) or {}).get("rc_full_pool", float("nan"))
            rc_b = (per_unit.get(("bcrea_band", inst, s)) or {}).get("rc_full_pool", float("nan"))
            rc_st = (per_unit.get(("bcrea_strict", inst, s)) or {}).get("rc_full_pool", float("nan"))
            if rec is None:
                continue
            menu_rows.append({
                "instance": inst, "spread": spread, "n_target": n, "seed": s,
                "rc_menu": rec["rc_menu"], "rc_menu_lb": rec["rc_menu_lb"],
                "rc_nominal_pool": rc_n, "rc_band_pool": rc_b, "rc_strict_pool": rc_st,
                "menu_minus_nominal": rec["rc_menu"] - rc_n,
                "menu_minus_band": rec["rc_menu"] - rc_b,
                "n_unique_menu": rec["n_unique_menu"],
            })
    write_rows(out / "v3_reopt_menu.csv",
               ["instance", "spread", "n_target", "seed", "rc_menu", "rc_menu_lb",
                "rc_nominal_pool", "rc_band_pool", "rc_strict_pool", "menu_minus_nominal",
                "menu_minus_band", "n_unique_menu"], menu_rows)

    # ================= 报告
    write_reports(out, cov_rows, rows_nom, rows_deg, rows_rc, rows_grid, rows_phen,
                  phen_scale, rows_sum, stats_rows, rows_reopt=reopt_rows, menu_rows=menu_rows,
                  n_files={"pools": len([p for p in POOL_DIR.glob("*.npz")]),
                           "scans": len([p for p in SCAN_DIR.glob("*.json")]),
                           "reopt": len([p for p in REOPT_DIR.glob("*.npz")])})
    print(f"全部完成，用时 {time.time()-t_start:.1f}s", flush=True)


# ---------------------------------------------------------------- 报告
def write_reports(out: Path, cov_rows, rows_nom, rows_deg, rows_rc, rows_grid, rows_phen,
                  phen_scale, rows_sum, stats_rows, rows_reopt, menu_rows, n_files) -> None:
    lines = ["v3 批次覆盖率与一致性核对报告（2026-10-19）", ""]
    lines.append(f"数据目录（只读）：{POOL_DIR.parent}")
    lines.append(f"文件数：pools={n_files['pools']}，scans={n_files['scans']}，reopt={n_files['reopt']}")
    lines.append("")
    by_kind = defaultdict(list)
    for r in cov_rows:
        by_kind[r["kind"]].append(r)
    for kind in ("pool", "scan", "reopt"):
        rr = by_kind[kind]
        st = defaultdict(int)
        for r in rr:
            st[r["status"]] += 1
        lines.append(f"== {kind}：{len(rr)} 个文件，PASS={st['PASS']} WARN={st['WARN']} FAIL={st['FAIL']}")
        for c in CHECKS:
            vals = [r[c] for r in rr if r.get(c) is not None]
            if not vals:
                continue
            n_ok = sum(1 for v in vals if v)
            flag = "" if n_ok == len(vals) else f"  ← 失败 {len(vals)-n_ok}"
            lines.append(f"   {c:20s} {n_ok}/{len(vals)}{flag}")
        bad = [r for r in rr if r["status"] != "PASS"]
        if bad:
            lines.append("   异常清单（最多 20 条）：")
            for r in bad[:20]:
                lines.append(f"     {r['file']} [{r['status']}] {r['note']}")
        lines.append("")
    lines.append("说明：pool 的 pool_size_ok 核对池规模=objs 长度=config 的 pop_size（NSGA-II 用 front1_size）；")
    lines.append("scan 的 provenance_ok 核对 pool_provenance.pool_file 与池 config（method/seed/decode_budget/actual）逐项一致；")
    lines.append("nominal_match_ok 核对 scan.nominal.best 与 max(objs0) 及池 npz 的 max(objs)（pessimistic 除外，")
    lines.append("其 npz objs 是悲观问题值）；eq_ok 用 core.robust_metrics.equiv_set 复算等价带；struct_ok 用池 decs 复算结构互异数。")
    lines.append("")
    lines.append("== 六方法主指标（三规模中位数；主工作点 δ=0.3, Γ=2）")
    lines.append("规模  方法            名义最优   带内退化   RC5     整池RC   最优单解WC")
    for r in rows_sum:
        lines.append(f"{r['n_target']:>4}  {SHORT[r['method']]:14s} {r['best_nominal_median']:9.3f}"
                     f" {r['deg_in_band_median']:10.3f} {r['rc5_median']:8.3f}"
                     f" {r['rc_full_median']:8.3f} {r['best_single_wc_median']:10.3f}")
    lines.append("")
    lines.append("== 现象（名义池，分规模；Q1 秩相关 / top5 质量差）")
    for r in phen_scale:
        lines.append(f"N={r['n_target']}: 名义全并列比例={r['share_nominal_all_tied']:.3f}，"
                     f"spearman 中位={r['spearman_median']:.3f}，top5 Jaccard 中位={r['jaccard_top5_median']:.3f}，"
                     f"平均质量差中位={r['mean_qual_gap_median']:.4f}，RC 增益>0 比例={r['share_rc_gap_positive']:.3f}，"
                     f"rel_spread 中位={r['rel_spread_median']:.4f}，等价带占池比例中位={r['eq_frac_median']:.3f}")
    lines.append("")
    lines.append("== 16 格网格：band/strict vs nominal 逐格胜率（配对 300 单元）")
    idx = {(r["cell"], r["instance"], r["seed"], r["method"]): r for r in rows_grid}
    for variant in ("bcrea_band", "bcrea_strict"):
        lines.append(f"-- {SHORT[variant]} vs nominal")
        for ck in CELL_KEYS:
            pairs = [(idx[(ck, i, s, variant)], idx[(ck, i, s, "nominal")])
                     for (i, s) in sorted({(r["instance"], r["seed"]) for r in rows_grid})
                     if (ck, i, s, variant) in idx and (ck, i, s, "nominal") in idx]
            if not pairs:
                continue
            win_deg = float(np.mean([a["deg_in_band_mean"] < b["deg_in_band_mean"] for a, b in pairs]))
            win_rc = float(np.mean([a["rc_full_pool"] > b["rc_full_pool"] for a, b in pairs]))
            d_deg = med([a["deg_in_band_mean"] - b["deg_in_band_mean"] for a, b in pairs])
            d_rc = med([a["rc_full_pool"] - b["rc_full_pool"] for a, b in pairs])
            lines.append(f"   {ck}: 退化胜率={win_deg:.3f} Δdeg中位={d_deg:+.4f}；"
                         f"RC 胜率={win_rc:.3f} ΔRC中位={d_rc:+.4f}")
        lines.append("")
    (out / "coverage_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- 配对统计摘要
    lines = ["v3 配对统计摘要（Wilcoxon 符号秩，双侧；Holm 在每个 subset|metric|variant 族内校正）", ""]
    lines.append("subset metric              variant       opponent     n   胜负/并列  Δ中位     p       p_holm  r_rb")
    for r in stats_rows:
        if r["subset"] != "all":
            continue
        lines.append(f"{r['subset']:6s} {r['metric']:17s} {SHORT[r['variant']]:12s} "
                     f"{SHORT[r['opponent']]:11s} {r['n_pairs']:3d} "
                     f"{r['wins']:3d}/{r['losses']:3d}/{r['ties']:3d} "
                     f"{r['median_diff']:+8.4f} {r['p_value']:.2e} {r['p_holm']:.2e} "
                     f"{r['rank_biserial']:+.3f}")
    (out / "v3_stats_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- reopt 摘要
    lines = ["v3 角点重解（66 个 Γ=2 分量角点，δ=0.3，6 实例 × 5 种子 = 1980）", ""]
    if rows_reopt:
        share = float(np.mean([r["gap_corner_minus_nominal"] >= 0 for r in rows_reopt]))
        lines.append(f"全部角点：角点最优 ≥ 名义池最优 的比例 = {share:.4f}；"
                     f"名义代价相对差中位 = {med([r['nominal_cost_rel'] for r in rows_reopt]):.4f}；"
                     f"elapsed 中位 = {med([r['elapsed_s'] for r in rows_reopt]):.1f}s")
        for inst in sorted({r["instance"] for r in rows_reopt}):
            rr = [r for r in rows_reopt if r["instance"] == inst]
            lines.append(f"  {inst:16s}: ≥ 比例={np.mean([r['gap_corner_minus_nominal'] >= 0 for r in rr]):.3f}，"
                         f"代价相对差中位={med([r['nominal_cost_rel'] for r in rr]):.4f}，"
                         f"角点最优中位={med([r['best_corner'] for r in rr]):.3f}")
    lines.append("")
    if menu_rows:
        lines.append("角点专才菜单（每角点取该角点池内最优解）整菜单 RC 与同实例同种子解池对比：")
        lines.append("  注意：菜单由 66 次独立重解组成（成本 66 × 9000N 解码），与单池（9000N）不是等预算对比。")
        lines.append("  RC_menu 为精确交叉评估值（66 专才 × 66 角点公共网格）；RC_menu_lb 为只用自己的角点值下界。")
        for n in (24, 48, 100):
            rr = [r for r in menu_rows if r["n_target"] == n]
            if not rr:
                continue
            lines.append(f"  N={n}: RC_menu 中位={med([r['rc_menu'] for r in rr]):.3f}，"
                         f"RC_lb 中位={med([r['rc_menu_lb'] for r in rr]):.3f}，"
                         f"nominal 池 RC 中位={med([r['rc_nominal_pool'] for r in rr]):.3f}，"
                         f"band 池 RC 中位={med([r['rc_band_pool'] for r in rr]):.3f}，"
                         f"strict 池 RC 中位={med([r['rc_strict_pool'] for r in rr]):.3f}")
            lines.append(f"        菜单 RC > nominal 池 RC：胜 {np.mean([r['menu_minus_nominal'] > 1e-9 for r in rr]):.2f} / "
                         f"平 {np.mean([abs(r['menu_minus_nominal']) <= 1e-9 for r in rr]):.2f}；"
                         f"菜单 RC > band 池 RC：胜 {np.mean([r['menu_minus_band'] > 1e-9 for r in rr]):.2f} / "
                         f"平 {np.mean([abs(r['menu_minus_band']) <= 1e-9 for r in rr]):.2f}；"
                         f"菜单唯一解数中位={med([r['n_unique_menu'] for r in rr]):.0f}/66")
    (out / "v3_reopt_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines[:10]))


if __name__ == "__main__":
    main()
