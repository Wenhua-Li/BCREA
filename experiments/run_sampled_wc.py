"""外部基线：采样鲁棒优化（scenario approach / robust counterpart 式鲁棒 EA）。

鲁棒优化文献的标准做法：适应度 = 不确定集合采样 M 个场景上的**最坏目标值**，
直接进化鲁棒解（不含名义目标、不含带机制）——回答审稿人"标准鲁棒 EA 不就够了？"。
与 BCREA 的区别：BCREA 保持名义收敛、只在名义等价带内引入鲁棒信号；
本基线把全部选择压力放在最坏情形上（预期：鲁棒但名义 visibly 差、带内密度低）。

算子复用 core.kcea；FE 口径默认 1000·N（论文口径），总解码量=(M+1)×FE，双口径报告。

服务器任务行样例：
  PYTHONPATH=. $PYTHON experiments/run_sampled_wc.py --scenario data/scenarios/tevc/Info_enemy_1.mat \
      --seed 1 --fes 1000x --m-scen 16 --delta 0.3 --gamma 2 --out <结果目录>/robwc_i1_s01.npz
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import core.robust_metrics as r6
from core.bcrea import BCREAConfig, build_scen_problems, scen_masks
from core.kcea import Population, environmental_selection, init_population, tournament_select, variation
from core.model import DefenseParams, SWTAProblem
from core.scenarios import load_enemy


def wc_fitness(decs: np.ndarray, objs_nom: np.ndarray, scen_probs: list[SWTAProblem],
               backend: str = "cpu") -> np.ndarray:
    """适应度 = M 场景最坏目标值（越大越鲁棒）。"""
    wc = np.full(len(decs), np.inf)
    for prob in scen_probs:
        wc = np.minimum(wc, r6.eval_pool(prob, decs, backend=backend))
    return wc


def run_sampled_wc(nominal: SWTAProblem, scen_probs: list[SWTAProblem], seed: int,
                   max_fes: int, pop_size: int | None = None, cfg: BCREAConfig = BCREAConfig(),
                   decode_budget: int | None = None, backend: str = "cpu"):
    """采样最坏情形主循环。返回 (末代种群, 鲁棒适应度, trace, 用时, cfgdict)。

    预算口径（与 run_bcrea 对齐）：真实解码量 = 初始 pop*(1+M) + 每代 off*(1+M)
    + 结尾名义复核 pop（自检用，预留不进搜索预算）；给了 decode_budget 就按它卡终止
    （下一代会超支就停，初始化计入预算），否则按名义 FE（--fes）旧行为。
    """
    rng = np.random.default_rng(seed)
    pop_size = pop_size or min(2 * nominal.n_target, 500)
    t0 = time.perf_counter()
    pop = init_population(nominal, pop_size, rng, backend=backend)
    fit = wc_fitness(pop.decs, pop.objs, scen_probs, backend=backend)
    n_evals = pop_size * (1 + len(scen_probs))
    fe = pop_size
    trace = [float(fit.max())]
    report_cost = pop_size   # 结尾名义复核的解码量，先从预算里扣出来
    search_budget = decode_budget - report_cost if decode_budget is not None else None
    while True:
        if search_budget is not None:
            if n_evals + pop_size * (1 + len(scen_probs)) > search_budget:
                break
        elif fe >= max_fes:
            break
        mating = tournament_select(-fit, 2, len(pop), rng)   # 最坏目标大者胜
        off = variation(pop.select(mating), nominal, rng, backend=backend)
        fe += len(pop)
        fit_off = wc_fitness(off.decs, off.objs, scen_probs, backend=backend)
        n_evals += len(off) * (1 + len(scen_probs))
        # matlab 式 1-1 替换，但比的是鲁棒适应度
        replace = fit <= fit_off
        new = pop.select(np.arange(len(pop)))
        idx = np.nonzero(replace)[0]
        if len(idx):
            new.decs[idx] = off.decs[idx]
            new.objs[idx] = off.objs[idx]
            new.kill_prob[idx] = off.kill_prob[idx]
            new.fail[idx] = off.fail[idx]
            for w in idx:
                new.kdg[w] = off.kdg[w]
                new.tasks[w] = off.tasks[w]
        # 缓存一致性：未被替换的行保留父代适应度（旧版整包 fit=fit_off，
        # 未替换父代的缓存对应另一份编码——评审 GPT-B3）
        fit = np.where(replace, fit_off, fit)
        pop = new
        trace.append(float(fit.max()))

    objs0 = r6.eval_pool(nominal, pop.decs, backend=backend)   # 名义目标复核（自检；pop.objs 本来就是名义值）
    n_evals += report_cost
    budget = int(decode_budget) if decode_budget is not None else int(max_fes) * (1 + cfg.n_scen) + report_cost
    cfgdict = dict(method="sampled_worst_case", baseline="sampled_robust_wc",
                   seed=int(seed), max_fes=int(max_fes), pop_size=int(pop_size),
                   m_scen=cfg.n_scen, delta=cfg.delta, gamma=cfg.gamma,
                   scen_seed=cfg.scen_seed, stratified=cfg.stratified,
                   total_evals=int(n_evals), actual_decode_evals=int(n_evals),
                   nominal_fes=int(fe), decode_budget=budget,
                   decode_budget_remaining=budget - int(n_evals), backend=backend,
                   train_masks=scen_masks(cfg, r6.n_components(nominal.params)).astype(int).tolist())
    return pop, objs0, fit, np.array(trace), time.perf_counter() - t0, cfgdict


def main() -> None:
    ap = argparse.ArgumentParser(description="采样鲁棒优化基线（worst-case over M scenarios）")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fes", default="1000x")
    ap.add_argument("--decode-budget", type=int, default=None,
                    help="真实解码总预算（v3 统一口径）；给了就按它卡终止，--fes 仅作名义参考")
    ap.add_argument("--m-scen", type=int, default=16)
    ap.add_argument("--delta", type=float, default=0.30)
    ap.add_argument("--gamma", type=int, default=2)
    ap.add_argument("--scen-seed", type=int, default=9001)
    ap.add_argument("--no-stratified", action="store_true", help="评分场景改随机采样（默认分层）")
    ap.add_argument("--backend", choices=["cpu", "gpu", "auto"], default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    backend = r6.resolve_backend(args.backend)

    cfg = BCREAConfig(delta=args.delta, gamma=args.gamma, n_scen=args.m_scen,
                      scen_seed=args.scen_seed, stratified=not args.no_stratified)
    enemy = load_enemy(args.scenario)
    base = DefenseParams()
    nominal = SWTAProblem(enemy, base)
    n_t = nominal.n_target
    scen_probs = build_scen_problems(enemy, base, cfg)
    max_fes = {"100x": 100 * n_t, "1000x": 1000 * n_t}.get(args.fes) or int(args.fes)

    pop, objs0, fit, trace, elapsed, cfgd = run_sampled_wc(
        nominal, scen_probs, seed=args.seed, max_fes=max_fes, cfg=cfg,
        decode_budget=args.decode_budget, backend=backend)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, decs=pop.decs, objs=objs0, fit_wc=fit, trace=trace,
        config=json.dumps({**cfgd, "scenario": str(args.scenario), "n_target": n_t}),
    )
    print(f"robwc 池完成：N={n_t} 池={len(pop)} 鲁棒适应度最优={fit.max():.4f} 名义最优={objs0.max():.4f} "
          f"真实解码={cfgd['actual_decode_evals']}/{cfgd['decode_budget']} "
          f"用时={elapsed:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
