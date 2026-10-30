# -*- coding: utf-8 -*-
"""v3 阶段1-C：分阶段墙钟计时（本机实测，全部实测不外推）。

五段（口径写死，见每段 note）：
  1. lookup_switch    给定角点，从 66×m 质量表取 argmax（纯查表；内层重复 1000 次/样本）
  2. menu_extract     core.robust_metrics.extract_menu 一次调用（δ=0.3, Γ=2，m∈{1,3,5}）
  3. quality_parse    读取并解析已存档 scan JSON（16 格质量矩阵；I/O + json 解析）
     quality_objsS    从已解析 JSON 取 d30_g2.objsS 并转 np.float64 数组
  4. milp_certify     exact_mstar 一次调用（MILP/HiGHS + 贪心停止数）
  5. corner_reopt     单次角点重解墙钟：直接统计 reopt npz 的 config.elapsed_s（不重跑）
  另附 build_reference：从扰动场景建质量矩阵（scan_cell，逐场景解码）——这是线上原始口径，
  只在 24 目标上测 1 个角点规模，供对照"读档 vs 现算"的差距（可选，默认跳过）。

输出：
  v3_timing.csv      segment, detail, n_target, n_samples, inner_reps, median/q25/q75/iqr/min/max, note
  v3_timing_env.txt  机器与软件环境、测量时的 CPU 负载、口径说明

跑法（仓库根；建议机器空闲时跑）：
  py -3.11 -X utf8 analysis/analyze_v3_timing.py
  py -3.11 -X utf8 analysis/analyze_v3_timing.py --build-reference   # 加测现算质量矩阵
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))   # 项目根（core 包）

import core.robust_metrics as r6  # noqa: E402
from analyze_v3 import (  # noqa: E402
    DEFAULT_OUT, METHODS, POOL_DIR, REOPT_DIR, REOPT_INSTANCES, SCAN_DIR, scenario_path,
)
from analyze_v3_mstar import exact_mstar  # noqa: E402

MENU_SIZES = [1, 3, 5]
GAMMAS = [1, 2, 3, 12]
LOOKUP_REPS = 1000
SEEDS = list(range(1, 6))


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


def summarize(vals: list[float], segment: str, detail: str, n_target, inner_reps: int,
              note: str, cpus: list[float] | None = None) -> dict:
    a = np.asarray(vals, float)
    row = {
        "segment": segment, "detail": detail, "n_target": n_target,
        "n_samples": int(len(a)), "inner_reps": inner_reps,
        "median_s": float(np.median(a)), "q25_s": float(np.percentile(a, 25)),
        "q75_s": float(np.percentile(a, 75)),
        "iqr_s": float(np.percentile(a, 75) - np.percentile(a, 25)),
        "min_s": float(a.min()), "max_s": float(a.max()),
        "median_cpu_s": None, "iqr_cpu_s": None, "note": note,
    }
    if cpus:
        c = np.asarray(cpus, float)
        row["median_cpu_s"] = float(np.median(c))
        row["iqr_cpu_s"] = float(np.percentile(c, 75) - np.percentile(c, 25))
    return row


def cpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
            text=True, timeout=20)
        name = " ".join(out.split())
        if name:
            return name
    except Exception:
        pass
    return platform.processor() or platform.machine()


def cpu_load_pct() -> str:
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage "
             "-Average).Average"],
            text=True, timeout=20)
        return out.strip()
    except Exception:
        return "unknown"


def ext_python_count() -> str:
    ps = ("(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "Where-Object { $_.CommandLine -notlike '*analyze_v3_timing*' }).Count")
    try:
        out = subprocess.check_output(["powershell", "-NoProfile", "-Command", ps],
                                      text=True, timeout=20)
        return out.strip() or "0"
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description="v3 分阶段墙钟计时")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--build-reference", action="store_true",
                    help="加测 scan_cell 现算质量矩阵（24 目标 1 个角点，慢）")
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--per-scale", type=int, default=3,
                    help="每规模每方法抽几个 (实例,种子)（默认 3 → 每规模 18 样本，三规模 54）")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rows: list[dict] = []
    t_all = time.time()

    # 预热（避免首次导入/内存分配计入第一个样本）
    _js0 = json.loads((SCAN_DIR / f"nominal_{REOPT_INSTANCES[0]}_s01.json")
                      .read_text(encoding="utf-8"))
    _S0 = np.asarray(_js0["cells"]["d30_g2"]["objsS"], float)
    _ = exact_mstar(_S0[:, np.asarray(_js0["nominal"]["eq_idx"], int)])

    scales = [24, 48, 100]
    per = args.per_scale

    def sample_units(n: int, k: int) -> list[tuple[str, str, int]]:
        """每方法抽 k 个 (实例,种子)，保证 6 方法都被覆盖。"""
        out = []
        insts = [i for i in REOPT_INSTANCES if int(i.split("_")[0][3:]) == n]
        for m in METHODS:
            out += [(m, rng.choice(insts), rng.choice(SEEDS)) for _ in range(k)]
        return out

    # ---------- 1. 查表换案（每规模 6 方法 × per 个样本）
    for n in scales:
        vals, cpus = [], []
        for method, inst, s in sample_units(n, per):
            m = rng.choice(MENU_SIZES)
            js = json.loads((SCAN_DIR / f"{method}_{inst}_s{s:02d}.json")
                            .read_text(encoding="utf-8"))
            decs = np.asarray(np.load(POOL_DIR / f"{method}_{inst}_s{s:02d}.npz",
                                      allow_pickle=False)["decs"])
            menu = r6.extract_menu(js, decs, m=m)
            S = np.asarray(js["cells"]["d30_g2"]["objsS"], float)
            Q = S[:, np.asarray(menu["members"], int)]      # 66×m 质量表
            k = rng.randrange(Q.shape[0])
            c0 = time.process_time()
            t0 = time.perf_counter()
            for _rep in range(LOOKUP_REPS):
                _ = int(np.argmax(Q[k]))
            vals.append((time.perf_counter() - t0) / LOOKUP_REPS)
            cpus.append((time.process_time() - c0) / LOOKUP_REPS)
        rows.append(summarize(vals, "lookup_switch", "argmax over menu (66×m table)", n,
                              LOOKUP_REPS,
                              "纯查表：Q[k].argmax()，内层 1000 次/样本；不含菜单提取与读档",
                              cpus))

    # ---------- 2. 菜单提取（每规模 6 方法 × m∈{1,3,5} 各一次）
    for n in scales:
        vals, cpus = [], []
        metas = []
        for (method, inst, s), m in zip(sample_units(n, per),
                                        MENU_SIZES * len(METHODS)):
            js = json.loads((SCAN_DIR / f"{method}_{inst}_s{s:02d}.json")
                            .read_text(encoding="utf-8"))
            decs = np.asarray(np.load(POOL_DIR / f"{method}_{inst}_s{s:02d}.npz",
                                      allow_pickle=False)["decs"])
            c0 = time.process_time()
            t0 = time.perf_counter()
            menu = r6.extract_menu(js, decs, m=m)
            vals.append(time.perf_counter() - t0)
            cpus.append(time.process_time() - c0)
            metas.append((m, len(js["nominal"]["eq_idx"]), len(decs)))
        rows.append(summarize(vals, "menu_extract",
                              "extract_menu (greedy RC + structural tie-break)", n, 1,
                              f"m∈{{1,3,5}}；样本 eq 中位="
                              f"{int(np.median([x[1] for x in metas]))}，池中位="
                              f"{int(np.median([x[2] for x in metas]))}；不含读档", cpus))

    # ---------- 3. 质量矩阵读档/解析（每规模 6 方法 × per 个文件）
    for n in scales:
        files = [SCAN_DIR / f"{m}_{i}_s{s:02d}.json" for m, i, s in sample_units(n, per)]
        vals_parse, vals_np, sizes_mb, targets = [], [], [], []
        cpus_parse, cpus_np = [], []
        vals_build, backends = [], []
        for fp in files:
            c0 = time.process_time()
            t0 = time.perf_counter()
            js = json.loads(fp.read_text(encoding="utf-8"))
            vals_parse.append(time.perf_counter() - t0)
            cpus_parse.append(time.process_time() - c0)
            c0 = time.process_time()
            t0 = time.perf_counter()
            S = np.asarray(js["cells"]["d30_g2"]["objsS"], float)
            vals_np.append(time.perf_counter() - t0)
            cpus_np.append(time.process_time() - c0)
            sizes_mb.append(fp.stat().st_size / 1e6)
            targets.append(S.shape[1])
            vals_build.append(float(js["cells"]["d30_g2"]["elapsed_s"]))
            backends.append(str((js.get("config") or {}).get("backend", "?")))
        rows.append(summarize(vals_parse, "quality_parse", "json.loads(scan file)", n, 1,
                              f"整份扫描 JSON（16 格）；文件中位={np.median(sizes_mb):.2f} MB；"
                              f"池规模中位={int(np.median(targets))}", cpus_parse))
        rows.append(summarize(vals_np, "quality_objsS", "np.asarray(cells.d30_g2.objsS)",
                              n, 1, "已解析 JSON → float64 矩阵（66×池规模）", cpus_np))
        rows.append(summarize(vals_build, "quality_build_archive",
                              "scan cell elapsed_s (66 scen × pool decode)", n, 1,
                              "存档实测：扰动问题构建 + 66×池规模解码；"
                              f"backend={sorted(set(backends))}；非本次重跑"))

    # ---------- 4. MILP 认证（每规模 6 方法 × Γ∈{1,2,3,12} 各一次 = 24 样本）
    for n in scales:
        vals, cpus = [], []
        metas = []
        insts = [i for i in REOPT_INSTANCES if int(i.split("_")[0][3:]) == n]
        for method in METHODS:
            for g in GAMMAS:
                inst, s = rng.choice(insts), rng.choice(SEEDS)
                js = json.loads((SCAN_DIR / f"{method}_{inst}_s{s:02d}.json")
                                .read_text(encoding="utf-8"))
                eq = np.asarray(js["nominal"]["eq_idx"], int)
                S = np.asarray(js["cells"][f"d30_g{g}"]["objsS"], float)[:, eq]
                c0 = time.process_time()
                t0 = time.perf_counter()
                ex, gr, rc, ok = exact_mstar(S)
                vals.append(time.perf_counter() - t0)
                cpus.append(time.process_time() - c0)
                metas.append((g, len(eq), ex, gr))
        rows.append(summarize(vals, "milp_certify",
                              "exact_mstar (MILP/HiGHS + greedy stop)", n, 1,
                              "Γ=1/2/3/12 各 6 方法；样本等价带中位="
                              f"{int(np.median([x[1] for x in metas]))}，m* 中位="
                              f"{np.median([x[2] for x in metas]):.0f}；不含读档", cpus))

    # ---------- 4b. 主档补测：Γ=2 的 MILP、m=5 的菜单提取（各规模 10 次）
    vals, cpus = [], []
    for n in scales:
        insts = [i for i in REOPT_INSTANCES if int(i.split("_")[0][3:]) == n]
        for _ in range(10):
            method = rng.choice(METHODS)
            inst, s = rng.choice(insts), rng.choice(SEEDS)
            js = json.loads((SCAN_DIR / f"{method}_{inst}_s{s:02d}.json")
                            .read_text(encoding="utf-8"))
            eq = np.asarray(js["nominal"]["eq_idx"], int)
            S = np.asarray(js["cells"]["d30_g2"]["objsS"], float)[:, eq]
            c0 = time.process_time()
            t0 = time.perf_counter()
            _ = exact_mstar(S)
            vals.append(time.perf_counter() - t0)
            cpus.append(time.process_time() - c0)
    rows.append(summarize(vals, "milp_certify", "exact_mstar, Γ=2 main cell (pooled)", None, 1,
                          "三规模各 10 次；主工作点认证成本", cpus))

    vals, cpus = [], []
    for n in scales:
        insts = [i for i in REOPT_INSTANCES if int(i.split("_")[0][3:]) == n]
        for _ in range(10):
            method = rng.choice(METHODS)
            inst, s = rng.choice(insts), rng.choice(SEEDS)
            js = json.loads((SCAN_DIR / f"{method}_{inst}_s{s:02d}.json")
                            .read_text(encoding="utf-8"))
            decs = np.asarray(np.load(POOL_DIR / f"{method}_{inst}_s{s:02d}.npz",
                                      allow_pickle=False)["decs"])
            c0 = time.process_time()
            t0 = time.perf_counter()
            _ = r6.extract_menu(js, decs, m=5)
            vals.append(time.perf_counter() - t0)
            cpus.append(time.process_time() - c0)
    rows.append(summarize(vals, "menu_extract", "extract_menu, m=5 (pooled)", None, 1,
                          "三规模各 10 次；认证菜单 m=5 的提取成本", cpus))

    # ---------- 5. 单次角点重解（读存档 elapsed_s）
    by_n = {}
    for inst in REOPT_INSTANCES:
        n = int(inst.split("_")[0][3:])
        for s in SEEDS:
            for fp in REOPT_DIR.glob(f"reopt_c*_{inst}_s{s:02d}.npz"):
                z = np.load(fp, allow_pickle=False)
                cfg = json.loads(str(z["config"]))
                by_n.setdefault(n, []).append(float(cfg["elapsed_s"]))
    for n in sorted(by_n):
        rows.append(summarize(by_n[n], "corner_reopt", f"reopt npz elapsed_s (N={n})",
                              n, 1,
                              "存档实测墙钟（KCEA 9000N FE，单角点）；非本次重跑"))
    all_e = [v for n in by_n for v in by_n[n]]
    rows.append(summarize(all_e, "corner_reopt", "reopt npz elapsed_s (all scales)",
                          None, 1, "N=24/48/100 混合"))

    # ---------- 附：现算质量矩阵（可选）
    if args.build_reference:
        from core.model import DefenseParams
        from core.scenarios import load_enemy
        inst = REOPT_INSTANCES[0]
        enemy = load_enemy(str(ROOT / scenario_path(inst)))
        decs = np.asarray(np.load(POOL_DIR / f"nominal_{inst}_s01.npz",
                                  allow_pickle=False)["decs"])[:24]
        t0 = time.perf_counter()
        masks, S, _dt = r6.scan_cell(enemy, DefenseParams(), decs, 0.3, 2, 12, 6000, 250,
                                     backend="cpu")
        dt = time.perf_counter() - t0
        rows.append({
            "segment": "build_reference", "detail": "scan_cell Γ=2 (24 decs, 66 scen)",
            "n_target": int(inst.split("_")[0][3:]), "n_samples": 1, "inner_reps": 1,
            "median_s": dt, "q25_s": np.nan, "q75_s": np.nan, "iqr_s": np.nan,
            "min_s": dt, "max_s": dt, "median_cpu_s": None, "iqr_cpu_s": None,
            "note": "单次现算（扰动问题构建+逐场景解码），仅 1 个样本，不作统计推断",
        })

    n1 = write_rows(out / "v3_timing.csv",
                    ["segment", "detail", "n_target", "n_samples", "inner_reps",
                     "median_s", "q25_s", "q75_s", "iqr_s", "min_s", "max_s",
                     "median_cpu_s", "iqr_cpu_s", "note"], rows)
    print(f"v3_timing.csv 写 {n1} 行", flush=True)

    env = [
        "v3 计时环境（2026-10-19）",
        f"CPU: {cpu_name()}",
        f"逻辑核数: {os.cpu_count()}",
        f"OS: {platform.platform()}",
        f"Python: {platform.python_version()}",
        f"numpy: {np.__version__}；scipy: {scipy.__version__}",
        f"测量时 CPU 平均负载: {cpu_load_pct()}%",
        f"测量时并发外部 Python 进程数: {ext_python_count()}（非本项目）",
        "",
        "口径说明：",
        "- lookup_switch：Q=S[:,members]（66×m）已建好，只计 Q[k].argmax()；内层 1000 次取均值。",
        "- menu_extract：只计 extract_menu 调用；JSON 读档与 npz 加载在计时外。",
        "- quality_parse：json.loads 整份 scans JSON（16 个 (δ,Γ) 格）；quality_objsS：转 float64。",
        "- quality_build_archive：直接取 scan JSON 的 cells.d30_g2.elapsed_s（扰动构建 + 66×池解码），",
        "  为战役存档实测（非本次重跑）；backend 见各行 note。",
        "- milp_certify：exact_mstar 一次调用（MILP + 贪心停止数）；读档在计时外。",
        "- corner_reopt：直接取 reopt npz 的 config.elapsed_s（KCEA，9000N FE），非本次重跑。",
        "- 每规模每段 n=18（6 方法 × 3 样本；菜单提取为 6 方法 × m∈{1,3,5}；",
        "  MILP 为 6 方法 × Γ∈{1,2,3,12}），三规模合计 54 个样本。",
        "- median_s 为墙钟（perf_counter）；median_cpu_s 为进程 CPU 时间（process_time），",
        "  在机器有外部负载时更接近隔离墙钟；两者接近说明测量时无抢占。",
        "- process_time 在本机分辨率约 15.6 ms，短操作（lookup/menu/objsS/milp）的 cpu 列记 0，",
        "  仅 quality_parse 的 cpu 列可用；短操作以墙钟为准。",
    ]
    (out / "v3_timing_env.txt").write_text("\n".join(env) + "\n", encoding="utf-8")
    print("\n".join(env[:6]), flush=True)
    print(f"全部完成，用时 {time.time() - t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
