"""MOEA 双目标基线池：NSGA-II（名义目标, 最坏退化）——回应"为什么不用多目标优化"。

目标：f1 = −名义目标（最大化期望摧毁）、f2 = K 个评分场景（与 BCREA 同一套，seed 9001）上的最坏退化。
算子复用 core.kcea（init_population/variation/Population），选择=非支配排序+拥挤距离（NSGA-II 标准）。
最终池 = 末代第一 Pareto 前沿成员（另存 decs_all=末代全种群供备选分析）。
与 BCREA 同 FE 口径（默认 1000·N）、同种群规模，公平对照。

服务器任务行样例：
  PYTHONPATH=. $PYTHON experiments/run_nsga2_pool.py --scenario data/scenarios/tevc/Info_enemy_1.mat \
      --seed 1 --fes 1000x --out <结果目录>/moea_i1_s01.npz
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import core.robust_metrics as r6
from core.bcrea import BCREAConfig, build_scen_problems, scen_masks, worst_deg
from core.kcea import Population, init_population, variation
from core.model import DefenseParams, SWTAProblem
from core.scenarios import load_enemy


def pareto_fronts(f1: np.ndarray, f2: np.ndarray) -> list[np.ndarray]:
    """两目标（都最小化）非支配分层：按 f1 排序扫 f2 运行最小值，逐层剥。返回下标数组列表。

    并列目标向量（f1、f2 均相等）互不支配，必须同层（评审 GPT-B3：旧版严格 `<` 把并列逐层拆开）。
    """
    idx = np.arange(len(f1))
    fronts = []
    alive = np.ones(len(f1), dtype=bool)
    while alive.any():
        cur = idx[alive]
        order = cur[np.lexsort((f2[cur], f1[cur]))]
        run_min = np.inf
        front = []
        for i in order:
            if f2[i] <= run_min + 1e-12:   # 含并列：f2 不严格更大即非被支配
                front.append(i)
                run_min = min(run_min, f2[i])
        fronts.append(np.array(front, dtype=int))
        alive[front] = False
    return fronts


def crowding(f1: np.ndarray, f2: np.ndarray, front: np.ndarray) -> np.ndarray:
    cd = np.zeros(len(front))
    for f in (f1, f2):
        v = f[front]
        order = np.argsort(v)
        cd[order[0]] = np.inf
        cd[order[-1]] = np.inf
        span = v[order[-1]] - v[order[0]]
        if span > 1e-12:
            cd[order[1:-1]] += (v[order[2:]] - v[order[:-2]]) / span
    return cd


def run_nsga2(nominal: SWTAProblem, scen_probs: list[SWTAProblem], seed: int,
              max_fes: int, pop_size: int | None = None, cfg: BCREAConfig = BCREAConfig(),
              decode_budget: int | None = None, backend: str = "cpu"):
    """NSGA-II 主循环。返回 (末代种群, deg, 第一前沿下标, 用时, cfgdict)。

    预算口径（与 run_bcrea 对齐）：真实解码量 = 初始 pop*(1+K) + 每代 off*(1+K)；
    给了 decode_budget 就按它卡终止（下一代会超支就停，初始化计入预算），
    否则按名义 FE（--fes）旧行为。
    """
    rng = np.random.default_rng(seed)
    pop_size = pop_size or min(2 * nominal.n_target, 500)
    t0 = time.perf_counter()
    pop = init_population(nominal, pop_size, rng, backend=backend)
    deg = worst_deg(pop.objs, pop.decs, scen_probs, backend=backend)
    n_evals = pop_size * (1 + len(scen_probs))
    fe = pop_size
    while True:
        if decode_budget is not None:
            if n_evals + pop_size * (1 + len(scen_probs)) > decode_budget:
                break
        elif fe >= max_fes:
            break
        # 父代选择：二元锦标赛比 (rank, -crowding)
        fronts = pareto_fronts(-pop.objs, deg)
        rank = np.empty(len(pop), dtype=int)
        cd_all = np.zeros(len(pop))
        for r, fr in enumerate(fronts):
            rank[fr] = r
            cd_all[fr] = crowding(-pop.objs, deg, fr)
        # 锦标赛：字典序比 (rank 升序, 拥挤距离降序)——不做数值归一化，
        # 拥挤距离含 inf（边界点），旧版 inf/inf=NaN 会污染 mating（评审 GPT-B3）
        a = rng.integers(0, len(pop), size=len(pop))
        b = rng.integers(0, len(pop), size=len(pop))
        a_wins = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (cd_all[a] > cd_all[b]))
        mating = np.where(a_wins, a, b)
        off = variation(pop.select(mating), nominal, rng, backend=backend)
        fe += len(pop)
        deg_off = worst_deg(off.objs, off.decs, scen_probs, backend=backend)
        n_evals += len(off) * (1 + len(scen_probs))

        # 合并 2N → 非支配排序 + 拥挤距离选 N
        all_decs = np.concatenate([pop.decs, off.decs])
        all_objs = np.concatenate([pop.objs, off.objs])
        all_deg = np.concatenate([deg, deg_off])
        all_kdg = list(pop.kdg) + list(off.kdg)
        all_tasks = list(pop.tasks) + list(off.tasks)
        all_kp = np.concatenate([pop.kill_prob, off.kill_prob])
        all_fail = np.concatenate([pop.fail, off.fail])
        fronts = pareto_fronts(-all_objs, all_deg)
        keep: list[int] = []
        for fr in fronts:
            if len(keep) + len(fr) <= pop_size:
                keep.extend(fr.tolist())
            else:
                cd = crowding(-all_objs, all_deg, fr)
                need = pop_size - len(keep)
                keep.extend(fr[np.argsort(-cd)[:need]].tolist())
                break
        keep = np.array(keep, dtype=int)
        pop = Population(decs=all_decs[keep].astype(np.int16), objs=all_objs[keep],
                         kdg=[all_kdg[i] for i in keep], tasks=[all_tasks[i] for i in keep],
                         kill_prob=all_kp[keep], fail=all_fail[keep])
        deg = all_deg[keep]

    front1 = pareto_fronts(-pop.objs, deg)[0]
    budget = int(decode_budget) if decode_budget is not None else int(max_fes) * (1 + cfg.n_scen)
    cfgdict = dict(method="nsga2", baseline="moea_nsga2", seed=int(seed), max_fes=int(max_fes),
                   pop_size=int(pop_size), front1_size=int(len(front1)),
                   delta=cfg.delta, gamma=cfg.gamma, n_scen=cfg.n_scen,
                   scen_seed=cfg.scen_seed, stratified=cfg.stratified,
                   total_evals=int(n_evals), actual_decode_evals=int(n_evals),
                   nominal_fes=int(fe), decode_budget=budget,
                   decode_budget_remaining=budget - int(n_evals), backend=backend,
                   train_masks=scen_masks(cfg, r6.n_components(nominal.params)).astype(int).tolist())
    if backend == "gpu":
        cfgdict["gpu_info"] = r6.gpu_info()
    return pop, deg, front1, time.perf_counter() - t0, cfgdict


def main() -> None:
    ap = argparse.ArgumentParser(description="MOEA 双目标基线（NSGA-II：名义 vs 最坏退化）")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fes", default="1000x")
    ap.add_argument("--decode-budget", type=int, default=None,
                    help="真实解码总预算（v3 统一口径）；给了就按它卡终止，--fes 仅作名义参考")
    ap.add_argument("--delta", type=float, default=0.30)
    ap.add_argument("--gamma", type=int, default=2)
    ap.add_argument("--n-scen", type=int, default=8)
    ap.add_argument("--scen-seed", type=int, default=9001)
    ap.add_argument("--no-stratified", action="store_true", help="评分场景改随机采样（默认分层）")
    ap.add_argument("--backend", choices=["cpu", "gpu", "auto"], default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    backend = r6.resolve_backend(args.backend)

    cfg = BCREAConfig(delta=args.delta, gamma=args.gamma, n_scen=args.n_scen,
                      scen_seed=args.scen_seed, stratified=not args.no_stratified)
    enemy = load_enemy(args.scenario)
    base = DefenseParams()
    nominal = SWTAProblem(enemy, base)
    n_t = nominal.n_target
    scen_probs = build_scen_problems(enemy, base, cfg)
    max_fes = {"100x": 100 * n_t, "1000x": 1000 * n_t}.get(args.fes) or int(args.fes)

    pop, deg, front1, elapsed, cfgd = run_nsga2(
        nominal, scen_probs, seed=args.seed, max_fes=max_fes, cfg=cfg,
        decode_budget=args.decode_budget, backend=backend)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, decs=pop.decs[front1], objs=pop.objs[front1], deg_wc=deg[front1],
        decs_all=pop.decs, objs_all=pop.objs, deg_all=deg,
        config=json.dumps({**cfgd, "scenario": str(args.scenario), "n_target": n_t}),
    )
    print(f"MOEA 池完成：N={n_t} 前沿={len(front1)}/{len(pop)} 名义最优={pop.objs.max():.4f} "
          f"前沿退化∈[{deg[front1].min():.3f},{deg[front1].max():.3f}] "
          f"真实解码={cfgd['actual_decode_evals']}/{cfgd['decode_budget']} "
          f"用时={elapsed:.0f}s -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
