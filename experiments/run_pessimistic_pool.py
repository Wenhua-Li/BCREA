"""悲观重跑基线池（红线#2 的"塌缩"情形，审稿人必问的对照）。

做法：把全部 12 个可行性分量同时缩水 δ（=Γ=全缩的"同一角点"悲观参数），
在该悲观问题上跑名义 KCEA（as-shipped 口径、FE=1000·N），存池 npz。
然后用 exp_robust 在**名义问题 + 全扰动网格**下评估这个池——
若"悲观重跑"的池在名义质量上大幅落后 BCREA、或鲁棒性并不更好，
则红线#2 担心的塌缩路径被实证排除，BCREA 的价值坐实。

服务器任务行样例：
  PYTHONPATH=. $PYTHON experiments/run_pessimistic_pool.py --scenario data/scenarios/tevc/Info_enemy_1.mat \
      --seed 1 --delta 0.3 --fes 1000x --out <结果目录>/pess_i1_s01.npz
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
    ap = argparse.ArgumentParser(description="悲观重跑基线池（全缩参数上的名义 KCEA）")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--delta", type=float, default=0.30)
    ap.add_argument("--fes", default="1000x")
    ap.add_argument("--backend", choices=["cpu", "gpu", "auto"], default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    backend = r6.resolve_backend(args.backend)

    enemy = load_enemy(args.scenario)
    base = DefenseParams()
    n_r = base.n_radar
    n_comp = r6.n_components(base)
    rad = np.full(n_r, 1.0 - args.delta)
    mis = np.full(n_comp - n_r, 1.0 - args.delta)
    pess = SWTAProblem(enemy, r6.perturbed_params(base, rad, mis))

    n_t = pess.n_target
    max_fes = resolve_fes(args.fes, n_t)
    # 悲观全缩后可能有目标完全打不到（v3 新实例实测 0~14 个），按发射失败处理继续搜索
    res = run_kcea(pess, seed=args.seed, max_fes=max_fes, infeasible_ok=True, backend=backend)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        decs=res.pop.decs, objs=res.pop.objs, trace=res.trace,
        config=json.dumps({**res.config, "method": "pessimistic_resolve",
                           "baseline": "pessimistic_rerun",
                           "delta": args.delta, "gamma": n_comp,
                           "components": list(range(n_comp)),
                           "elapsed_s": round(res.elapsed, 1),
                           "scenario": str(args.scenario),
                           "n_target": n_t,
                           "note": "objs 是悲观问题上的值；名义/扰动评估由 exp_robust 重算"}),
    )
    print(f"悲观池完成：N={n_t} 池={len(res.pop)} 悲观最优={res.pop.objs.max():.4f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
