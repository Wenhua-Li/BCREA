"""E1 角点重解基线（Bertsimas 家族的"角点曲目"对照）。

做法：把 --components 指定的能力分量子集缩水 δ（其余保持名义），在该角点问题上跑名义 KCEA
（FE=1000·N 论文口径）。12 个单分量角点各跑一遍后，由 make_union_pool.py 并成"角点曲目池"，
再用 exp_robust 在名义问题+扰动网格下评估——对照 BCREA 认证菜单的 RC 与密度生态位。

与 exp_pess_pool 的区别：pess 是全缩（Γ=C 塌缩角点），本脚本是任意子集（Γ=1 单分量角点为主）。
服务器任务行样例：
  PYTHONPATH=. $PYTHON experiments/run_corner_pool.py --scenario data/scenarios/tevc/Info_enemy_1.mat \
      --components 0 --seed 1 --delta 0.3 --fes 1000x --out <dir>/corner_c0_i1_s01.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import core.robust_metrics as r6
from core.kcea import resolve_fes, run_kcea
from core.model import DefenseParams, SWTAProblem
from core.scenarios import load_enemy


def main() -> None:
    ap = argparse.ArgumentParser(description="角点重解基线池（指定分量子集缩水上的名义 KCEA）")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--components", required=True,
                    help="缩水分量索引列表（0..n_r-1 为雷达，n_r..n_comp-1 为杀伤边界），逗号分隔")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--delta", type=float, default=0.30)
    ap.add_argument("--fes", default="1000x")
    ap.add_argument("--backend", choices=["cpu", "gpu", "auto"], default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    backend = r6.resolve_backend(args.backend)

    comps = [int(c) for c in args.components.split(",")]
    enemy = load_enemy(args.scenario)
    base = DefenseParams()
    n_r = base.n_radar
    n_comp = r6.n_components(base)
    assert all(0 <= c < n_comp for c in comps), f"分量索引越界：{comps}（n_comp={n_comp}）"
    scale = np.ones(n_comp)
    scale[comps] = 1.0 - args.delta
    corner = SWTAProblem(enemy, r6.perturbed_params(base, scale[:n_r], scale[n_r:]))

    n_t = corner.n_target
    max_fes = resolve_fes(args.fes, n_t)
    # 角点缩水后可能有目标完全打不到（v3 新实例实测最多 6 个），按发射失败处理继续搜索
    res = run_kcea(corner, seed=args.seed, max_fes=max_fes, infeasible_ok=True, backend=backend)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        decs=res.pop.decs, objs=res.pop.objs, trace=res.trace,
        config=json.dumps({**res.config, "method": "true_reoptimization",
                           "baseline": "corner_re-solve",
                           "components": comps, "delta": args.delta,
                           "gamma": len(comps),
                           "elapsed_s": round(res.elapsed, 1),
                           "scenario": str(args.scenario), "n_target": n_t,
                           "note": "objs 是角点问题上的值；名义/扰动评估由 exp_robust 重算"}),
    )
    print(f"角点池完成：N={n_t} comps={comps} 池={len(res.pop)} 角点最优={res.pop.objs.max():.4f} -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
