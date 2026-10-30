"""v3 正式实验：解池 × 预算式扰动扫描 → 鲁棒度量 JSON。

一次调用 = 一个 (场景, 解池) 跑给定的 δ 列表 × Γ 列表；同 (δ,Γ,场景文件,scen_seed)
的扰动场景采样跨解池种子确定可复现（core.robust_metrics.cell_seed），保证不同池面对同一批场景。

解池来源（二选一）：
  --pool  npz        复用已生成的池（同格式 npz）；
  --gen-pool-seed S  现场跑 KCEA 生成（--fes 默认 1000x 正式口径；开发冒烟可 100x）。

任务行样例（任务清单由 experiments/make_tasks.py 生成、experiments/local_run.py 执行，规矩见 AGENTS.md 第7条）：
  PYTHONPATH=. $PYTHON experiments/run_pool_scan.py \
      --scenario data/scenarios/tevc/Info_enemy_1.mat \
      --pool data/results/<pool批次>/pool_i1_s01.npz \
      --deltas 0.1,0.2,0.3,0.4 --gammas 1,2,3,12 --n-scen 60 \
      --out data/results/<tag>/e1_s01_g2.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import core.robust_metrics as r6
from core.kcea import run_kcea
from core.model import DefenseParams, SWTAProblem
from core.scenarios import load_enemy


def _py(x):
    """numpy → 纯 Python（JSON 可序列化），浮点保留 6 位。"""
    if isinstance(x, np.ndarray):
        return [round(float(v), 6) for v in x.ravel().tolist()]
    if isinstance(x, (np.floating, float)):
        return round(float(x), 6)
    if isinstance(x, (np.integer, int)):
        return int(x)
    raise TypeError(f"不会序列化 {type(x)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="解池×预算式扰动扫描")
    ap.add_argument("--scenario", required=True, help=".mat 场景文件（写相对路径，如 data/scenarios/tevc/...）")
    ap.add_argument("--pool", default=None, help="解池 npz（同格式）")
    ap.add_argument("--gen-pool-seed", type=int, default=None, help="无 npz 时现场生成解池的种子")
    ap.add_argument("--fes", default="1000x", help="现场生成解池的 FE 口径（正式=1000x）")
    ap.add_argument("--selection", default="matlab", choices=["matlab", "paper"])
    ap.add_argument("--deltas", default="0.1,0.2,0.3,0.4")
    ap.add_argument("--gammas", default="1,2,3,12")
    ap.add_argument("--n-scen", type=int, default=60, help="每 (δ,Γ) 单元的扰动场景数（穷举不可行时的采样数）")
    ap.add_argument("--enum-cap", type=int, default=250, help="C(分量数,Γ)≤此值时穷举全部场景（最坏情形精确）；0=强制采样")
    ap.add_argument("--scen-seed", type=int, default=6000, help="场景采样基础种子")
    ap.add_argument("--band-rel", type=float, default=0.005, help="名义等价带（相对最优）")
    ap.add_argument("--no-objsmatrix", action="store_true", help="JSON 不存完整 objsS 矩阵（省空间，但丢解集策略离线分析能力）")
    ap.add_argument("--backend", choices=["cpu", "gpu", "auto"], default="cpu")
    ap.add_argument("--gpu-batch", type=int, default=256)
    ap.add_argument("--out", required=True, help="输出 JSON 路径")
    args = ap.parse_args()

    deltas = [float(x) for x in args.deltas.split(",")]
    gammas = [int(x) for x in args.gammas.split(",")]

    enemy = load_enemy(args.scenario)
    base_params = DefenseParams()
    nominal = SWTAProblem(enemy, base_params)
    n_t = nominal.n_target
    backend = r6.resolve_backend(args.backend)

    # —— 解池 ——
    pool_provenance: dict = {}
    if args.pool:
        d = np.load(args.pool, allow_pickle=False)
        decs = d["decs"].astype(np.int16)
        pool_provenance = {"pool_file": str(args.pool), "config": json.loads(str(d["config"]))}
    elif args.gen_pool_seed is not None:
        max_fes = {"100x": 100 * n_t, "1000x": 1000 * n_t}.get(args.fes) or int(args.fes)
        t0 = time.perf_counter()
        res = run_kcea(nominal, seed=args.gen_pool_seed, max_fes=max_fes, selection=args.selection,
                       backend=backend)
        decs = res.pop.decs
        pool_provenance = {
            "gen_pool": {"seed": args.gen_pool_seed, "fes": max_fes, "selection": args.selection,
                         "elapsed_s": round(time.perf_counter() - t0, 1)},
        }
    else:
        raise SystemExit("必须给 --pool 或 --gen-pool-seed 之一")

    # 名义目标值：现场重算（同时充当流水线自检：应与池里存的 objs 一致）
    objs0 = r6.eval_pool(nominal, decs, backend=backend, batch_size=args.gpu_batch)
    eq_idx = r6.equiv_set(objs0, args.band_rel)
    best = float(objs0.max())
    spread_obj0 = float(objs0[eq_idx].max() - objs0[eq_idx].min())
    n_struct = r6.n_struct_distinct(decs, eq_idx)
    print(
        f"场景={Path(args.scenario).name} N={n_t} 池={len(decs)} best={best:.4f} "
        f"等价集={len(eq_idx)}(结构互异{n_struct}) 名义极差={spread_obj0:.4f}",
        flush=True,
    )

    out: dict = {
        "config": {
            "scenario": str(args.scenario), "n_target": n_t, "pool_size": len(decs),
            "deltas": deltas, "gammas": gammas, "n_scen": args.n_scen,
            "scen_seed": args.scen_seed, "band_rel": args.band_rel,
            "n_comp": r6.n_components(base_params),
            "backend": backend, "batch_size": args.gpu_batch, "gpu_info": r6.gpu_info(),
        },
        "pool_provenance": pool_provenance,
        "nominal": {
            "best": best, "objs0": _py(objs0), "eq_idx": _py(eq_idx),
            "spread_obj0_eq": spread_obj0, "n_struct_distinct_eq": n_struct,
        },
        "cells": {},
    }

    total_t0 = time.perf_counter()
    for delta in deltas:
        for gamma in gammas:
            masks, objsS, dt = r6.scan_cell(enemy, base_params, decs, delta, gamma, args.n_scen, args.scen_seed, args.enum_cap, backend=backend, batch_size=args.gpu_batch)
            m = r6.cell_metrics(objs0, objsS, eq_idx)
            cell = {
                "delta": delta, "gamma": gamma, "n_scen_used": int(objsS.shape[0]), "elapsed_s": round(dt, 1),
                "spread_wc": m["spread_wc"], "spread_deg": m["spread_deg"],
                "amp": m["amp"] if np.isfinite(m["amp"]) else None,
                "rel_spread": m["rel_spread"], "reorders": m["reorders"],
                "n_distinct_worst": m["n_distinct_worst"],
                "objs_wc": _py(m["objs_wc"]), "argmin": _py(m["argmin"]), "deg": _py(m["deg"]),
                # 解集层面（鲁棒覆盖）：整池 RC + 单解最坏目标的最大值，主结论分析用
                "rc_full_pool": r6.robust_coverage(objsS, np.arange(len(decs))),
                "best_single_wc": float(m["objs_wc"].max()),
            }
            if not args.no_objsmatrix:
                cell["masks"] = masks.astype(int).tolist()
                cell["objsS"] = [[round(float(v), 6) for v in row] for row in objsS]
            out["cells"][f"d{round(delta*100):02d}_g{gamma}"] = cell
            print(
                f"  δ={delta:.2f} Γ={gamma:>2}: {dt:6.1f}s spread_wc={m['spread_wc']:.4f} "
                f"amp={m['amp']:.2f} rel={m['rel_spread']*100:.2f}% 重排={m['reorders']} "
                f"不同最坏场景={m['n_distinct_worst']}/{len(eq_idx)} RC(整池)={cell['rc_full_pool']:.4f}",
                flush=True,
            )

    out["total_elapsed_s"] = round(time.perf_counter() - total_t0, 1)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"DONE {outp} 总用时={out['total_elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
