"""BCREA（Band-confined Coverage-driven Robustness-aware Evolutionary Algorithm）主算法。

口径：BCREA 是围绕"认证"目标设计的两段式算法，不是 KCEA 的改进版：
  - 段一搜索：带内鲁棒感知环境选择（BCREA 新组件）——名义等价带内按最坏退化择优替换；
  - 段二提取：覆盖式贪心提取（BCREA 新组件，见 core.robust_metrics.greedy_rc_select）——按鲁棒覆盖 RC
    从最终池抽出认证菜单。
  变异算子与构造式解码器沿用知识引导 SWTA 文献的实现（core.kcea 组件，原样复用）；
  锦标赛选择与名义目标评估也来自该组件。

设计依据（robust1 批 + 深化分析数据，见 PROGRESS 与 *_analysis/deep_summary.txt）：
  Q1 名义排名对鲁棒排名几乎零信息（Spearman≈0.1，100 目标名义全并列）→ 搜索必须显式引入鲁棒信号；
  Q3 结构多样性 ≠ 功能鲁棒覆盖 → 鲁棒评分用"采样扰动场景上的最坏退化"，不用结构距离；
  Q4 RC 为主、多样为辅几乎零成本 → 段一机制："名义带内比鲁棒"。

段一环境选择规则：
      子代名义 > 父代名义 + ε      → 替换（名义明显更好）
      |子代名义 − 父代名义| ≤ ε    → 比最坏退化：子代更鲁棒才替换（带内鲁棒者胜）
      父代名义 > 子代名义 + ε      → 保留父代
    ε = eps_band_rel × 当前最优名义（默认 0.5%，与名义等价集口径一致）。
  鲁棒评分 = K 个采样扰动场景（Γ 预算式，种子固定）上的最坏退化 deg = obj0 − min_s obj_s；
  每代只给子代新算，父代携带缓存。成本 ≈ (K+1)× 名义评估。

输出 npz 与 core.kcea 兼容（decs/objs/trace/config + 新增 deg_wc），
可直接喂 exp_robust.py --pool 做扰动扫描对比。

任务行样例（仓库根）：
  PYTHONPATH=. python core/bcrea.py --scenario data/scenarios/tevc/Info_enemy_1.mat \
      --seed 1 --fes 1000x --delta 0.3 --gamma 2 --n-scen 8 --out <结果目录>/bcrea_i1_s01.npz
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import core.robust_metrics as r6
from core.kcea import Population, init_population, tournament_select, variation
from core.model import DefenseParams, SWTAProblem
from core.scenarios import load_enemy


@dataclass
class BCREAConfig:
    delta: float = 0.30          # 评分场景的缩水幅度（主判据档）
    gamma: int = 2               # 评分场景的预算
    n_scen: int = 8              # 评分场景个数 K（成本 ≈ (K+1)× 名义）
    scen_seed: int = 9001        # 评分场景采样种子（跨 run 固定 → 同一套场景，公平可比）
    eps_band_rel: float = 0.005  # 名义等价带（与等价集口径一致）
    stratified: bool = True      # 评分场景分层采样（保证 12 分量全覆盖；修 seed 敏感性，v1 默认开；
    #                            #   sig/quality1-2 批为 False，config 存档可区分）
    no_downgrade: bool = False   # True=严格模式：带内比鲁棒仅当子代名义不降（防"带内向下漂移"，
    #                            #   冒烟实测 v0 宽松规则 500 代链式漂掉 0.51 名义）
    invert_fitness: bool = False


def build_scen_problems(enemy, base: DefenseParams, cfg: BCREAConfig) -> list[SWTAProblem]:
    """K 个评分用扰动问题（分层或随机采样自 Γ 预算空间；enum_cap=0 强制采样 k 个）。"""
    if cfg.stratified:
        masks = r6.stratified_masks(cfg.n_scen, cfg.gamma, cfg.scen_seed, r6.n_components(base))
    else:
        masks = r6.sample_masks(cfg.gamma, cfg.n_scen, cfg.scen_seed, r6.n_components(base), enum_cap=0)
    probs = []
    for mask in masks:
        rad, mis = r6.scales_from_mask(mask, cfg.delta, base.n_radar)
        probs.append(SWTAProblem(enemy, r6.perturbed_params(base, rad, mis)))
    return probs


def worst_deg(objs0: np.ndarray, decs: np.ndarray, scen_probs: list[SWTAProblem],
              backend: str = "cpu", batch_size: int = 256) -> np.ndarray:
    """最坏退化 deg = obj0 − min_s obj_s（每个个体在 K 个评分场景上的最坏情形）。"""
    wc = np.full(len(decs), np.inf)
    for prob in scen_probs:
        wc = np.minimum(wc, r6.eval_pool(prob, decs, backend=backend, batch_size=batch_size))
    return objs0 - wc


def selection_label(cfg: BCREAConfig) -> str:
    """输出口径标签：严格模式 bcrea_strict、宽松模式 bcrea_band（不再用含糊的 bcrea_v0）。"""
    return "bcrea_strict" if cfg.no_downgrade else "bcrea_band"


def scen_masks(cfg: BCREAConfig, n_comp: int) -> np.ndarray:
    """训练评分场景掩码（build_scen_problems 的同一来源；存档进 npz，保证可复现）。"""
    if cfg.stratified:
        return r6.stratified_masks(cfg.n_scen, cfg.gamma, cfg.scen_seed, n_comp)
    return r6.sample_masks(cfg.gamma, cfg.n_scen, cfg.scen_seed, n_comp, enum_cap=0)


def run_bcrea(problem: SWTAProblem, scen_probs: list[SWTAProblem], seed: int,
              max_fes: int, pop_size: int | None = None, cfg: BCREAConfig = BCREAConfig(),
              decode_budget: int | None = None, verbose: bool = False,
              backend: str = "cpu", batch_size: int = 256):
    """BCREA 段一（带内鲁棒感知搜索）主循环。返回 (pop, deg, trace_nom, trace_deg, elapsed, cfgdict)。

    预算口径（v3 统一）：
      - 给了 decode_budget：终止条件按真实解码量 n_evals 卡——下一代解码（含评分场景重估）
        会超支就停；初始化已计入预算；父代缓存不重估（维持原逻辑）。
      - 没给：保持旧的 --fes 行为（按名义 FE 计数），decode_budget 记 max_fes*(1+K)
        （旧任务的可预期真实解码上限），actual_decode_evals 记真实 n_evals。
    """
    rng = np.random.default_rng(seed)
    pop_size = pop_size or min(2 * problem.n_target, 500)
    t0 = time.perf_counter()

    pop = init_population(problem, pop_size, rng, backend=backend)
    deg = worst_deg(pop.objs, pop.decs, scen_probs, backend=backend, batch_size=batch_size)
    n_evals = len(pop) * (1 + len(scen_probs))   # 真实解码总量记账（评审 P4/B#3）
    fe = pop_size
    trace_nom = [float(pop.objs.max())]
    trace_deg = [float(deg.mean())]

    while True:
        # 终止条件：预算模式看真实解码量，旧模式看名义 FE
        if decode_budget is not None:
            if n_evals + len(pop) * (1 + len(scen_probs)) > decode_budget:
                break
        elif fe >= max_fes:
            break
        fit = -pop.objs if cfg.invert_fitness else pop.objs
        mating = tournament_select(fit, 2, len(pop), rng)
        off = variation(pop.select(mating), problem, rng, cfg.invert_fitness, backend=backend)
        fe += len(pop)
        deg_off = worst_deg(off.objs, off.decs, scen_probs, backend=backend, batch_size=batch_size)
        n_evals += len(off) * (1 + len(scen_probs))

        best = max(pop.objs.max(), off.objs.max())
        eps = cfg.eps_band_rel * best
        replace = np.zeros(len(pop), dtype=bool)
        for i in range(len(pop)):
            if off.objs[i] > pop.objs[i] + eps:
                replace[i] = True                      # 名义明显更好
            elif abs(off.objs[i] - pop.objs[i]) <= eps:
                # 带内：更鲁棒者胜（v0 核心机制）。严格模式要求名义不降，防链式向下漂移
                if deg_off[i] < deg[i] - 1e-9 and (not cfg.no_downgrade or off.objs[i] >= pop.objs[i]):
                    replace[i] = True

        new = pop.select(np.arange(len(pop)))          # 先整体拷贝父代
        idx = np.nonzero(replace)[0]
        if len(idx):
            new.decs[idx] = off.decs[idx]
            new.objs[idx] = off.objs[idx]
            new.kill_prob[idx] = off.kill_prob[idx]
            new.fail[idx] = off.fail[idx]
            for w in idx:
                new.kdg[w] = off.kdg[w]
                new.tasks[w] = off.tasks[w]
        new_deg = deg.copy()
        new_deg[idx] = deg_off[idx]
        pop, deg = new, new_deg

        trace_nom.append(float(pop.objs.max()))
        trace_deg.append(float(deg.mean()))
        if verbose and len(trace_nom) % 50 == 0:
            print(f"  FE={fe:>7} 名义最优={trace_nom[-1]:.4f} 平均最坏退化={trace_deg[-1]:.4f}", flush=True)

    budget = int(decode_budget) if decode_budget is not None else int(max_fes) * (1 + cfg.n_scen)
    return pop, deg, np.array(trace_nom), np.array(trace_deg), time.perf_counter() - t0, \
        dict(delta=cfg.delta, gamma=cfg.gamma, n_scen=cfg.n_scen, scen_seed=cfg.scen_seed,
             eps_band_rel=cfg.eps_band_rel, invert_fitness=cfg.invert_fitness,
             no_downgrade=cfg.no_downgrade, stratified=cfg.stratified,
             method=selection_label(cfg), selection=selection_label(cfg),
             total_evals=int(n_evals), actual_decode_evals=int(n_evals),
             nominal_fes=int(fe), decode_budget=budget,
             decode_budget_remaining=budget - int(n_evals),
             train_masks=scen_masks(cfg, r6.n_components(problem.params)).astype(int).tolist(),
             backend=backend, batch_size=int(batch_size), gpu_info=r6.gpu_info())


def main() -> None:
    ap = argparse.ArgumentParser(description="BCREA：带内鲁棒感知环境选择（段一搜索）")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fes", default="1000x", help="100x（dev）/ 1000x（正式口径）/ 具体数字")
    ap.add_argument("--decode-budget", type=int, default=None,
                    help="真实解码总预算（v3 统一口径）；给了就按它卡终止，--fes 仅作名义参考")
    ap.add_argument("--protocol", default=None,
                    help="实验协议标签（如 v3_equal_decode），给了就写进 config")
    ap.add_argument("--pop", type=int, default=None)
    ap.add_argument("--delta", type=float, default=0.30)
    ap.add_argument("--gamma", type=int, default=2)
    ap.add_argument("--n-scen", type=int, default=8)
    ap.add_argument("--scen-seed", type=int, default=9001)
    ap.add_argument("--eps-band", type=float, default=0.005)
    ap.add_argument("--no-stratified", action="store_true", help="评分场景改随机采样（复现 sig 批口径用；默认分层）")
    ap.add_argument("--no-downgrade", action="store_true",
                    help="严格模式：带内比鲁棒仅当子代名义不降（防向下漂移，名义零牺牲）")
    ap.add_argument("--backend", choices=["cpu", "gpu", "auto"], default="cpu")
    ap.add_argument("--gpu-batch", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    backend = r6.resolve_backend(args.backend)
    cfg = BCREAConfig(delta=args.delta, gamma=args.gamma, n_scen=args.n_scen,
                      scen_seed=args.scen_seed, eps_band_rel=args.eps_band,
                      stratified=not args.no_stratified,
                      no_downgrade=args.no_downgrade)
    enemy = load_enemy(args.scenario)
    base = DefenseParams()
    nominal = SWTAProblem(enemy, base)
    n_t = nominal.n_target
    max_fes = {"100x": 100 * n_t, "1000x": 1000 * n_t}.get(args.fes) or int(args.fes)

    t0 = time.perf_counter()
    scen_probs = build_scen_problems(enemy, base, cfg)
    print(f"评分场景：{len(scen_probs)} 个（δ={cfg.delta}, Γ={cfg.gamma}, seed={cfg.scen_seed}），"
          f"构建 {time.perf_counter()-t0:.1f}s", flush=True)

    pop, deg, trace_nom, trace_deg, elapsed, cfgd = run_bcrea(
        nominal, scen_probs, seed=args.seed, max_fes=max_fes, pop_size=args.pop,
        cfg=cfg, decode_budget=args.decode_budget, verbose=True,
        backend=backend, batch_size=args.gpu_batch)

    eq = r6.equiv_set(pop.objs, cfg.eps_band_rel)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg_out = {**cfgd, "seed": args.seed, "max_fes": max_fes,
               "pop_size": len(pop),
               "scenario": str(args.scenario), "n_target": n_t}
    if args.protocol:   # 协议标签（如 v3_equal_decode）只在命令行显式提供时入档
        cfg_out["protocol"] = args.protocol
    np.savez_compressed(
        out, decs=pop.decs, objs=pop.objs, deg_wc=deg,
        trace=trace_nom, trace_deg=trace_deg,
        config=json.dumps(cfg_out),
    )
    print(f"完成：N={n_t} 池={len(pop)} 名义最优={pop.objs.max():.4f} 等价集={len(eq)} "
          f"等价集平均退化={deg[eq].mean():.4f}（全池 {deg.mean():.4f}） "
          f"真实解码={cfgd['actual_decode_evals']}/{cfgd['decode_budget']} 用时={elapsed:.1f}s -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
