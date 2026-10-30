"""KCEA（前作 TEVC 2025 知识引导竞争进化算法）的 Python 移植。

前作代码与论文有四处不一致（2026-10-02 移植时核定，账本有记录）：
  1. FE 口径：代码 100·N，论文 1000·N —— 这里做成参数，正式实验一律 1000·N（项目口径）。
  2. 种群：代码 2·N 无上限，论文 min(2·N, 500) —— 做成参数，默认按论文。
  3. 环境选择：代码是"父子 1-1 精英替换"，其后的多样性修剪循环因种群永不超员而是死代码；
     论文 Algorithm 3 描述的是"父子合并 + 相似度竞争淘汰"。两种都实现：
     selection="matlab"（默认，忠实原码行为）/ "paper"（论文描述的竞争选择）。
  4. 锦标赛：原码把原始适应度直接喂给 PlatEMO 的最小者胜锦标赛（交配池偏好低适应度、
     知识引导变异偏好低概率组合）。这是前作数字的真实行为，默认照抄；
     需要"高者胜"时传 invert_fitness=True。

用法（库）：
    from core.kcea import run_kcea
    result = run_kcea(problem, seed=1, max_fes=1000*problem.n_target)

用法（命令行，服务器批量跑解池用）：
    python -m core.kcea --scenario Scenario/Info_enemy_1.mat --seed 1 \
        --fes 1000x --selection matlab --out results/pool.npz
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from core.decoder import EvalResult
from core.decoder_gpu import evaluate_batch, gpu_info, resolve_backend
from core.model import SWTAProblem, DefenseParams
from core.scenarios import load_enemy

# FE 口径映射（100x 开发冒烟 / 1000x 论文口径 / 9000x v3 统一解码预算口径），其余按整数解析
FE_MULT = {"100x": 100, "1000x": 1000, "9000x": 9000}


def resolve_fes(fes: str, n_target: int) -> int:
    """把 '100x'/'1000x'/'9000x' 或具体数字解析成 FE 上限。"""
    if fes in FE_MULT:
        return FE_MULT[fes] * n_target
    return int(fes)


@dataclass
class Population:
    decs: np.ndarray  # (n, K, N, 2) int16
    objs: np.ndarray  # (n,)
    kdg: list  # 每个个体的知识行 (m,2)
    tasks: list  # 每个个体的任务列表
    kill_prob: np.ndarray  # (n, N, K+1)
    fail: np.ndarray  # (n, N, K)

    def __len__(self) -> int:
        return len(self.objs)

    def select(self, idx: np.ndarray) -> "Population":
        idx = np.asarray(idx)
        return Population(
            decs=self.decs[idx],
            objs=self.objs[idx],
            kdg=[self.kdg[i] for i in idx],
            tasks=[self.tasks[i] for i in idx],
            kill_prob=self.kill_prob[idx],
            fail=self.fail[idx],
        )

    @classmethod
    def from_individuals(cls, decs: np.ndarray, evs: list[EvalResult]) -> "Population":
        return cls(
            decs=np.stack(decs).astype(np.int16),
            objs=np.array([ev.obj for ev in evs]),
            kdg=[ev.kdg for ev in evs],
            tasks=[ev.tasks for ev in evs],
            kill_prob=np.stack([ev.kill_prob for ev in evs]),
            fail=np.stack([ev.fail for ev in evs]),
        )


def tournament_select(fitness: np.ndarray, k: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """PlatEMO 锦标赛（最小者胜，稳定排名），前作 TournamentSelection.m 的移植。"""
    rank = np.argsort(np.argsort(fitness, kind="stable"), kind="stable")
    parents = rng.integers(0, len(fitness), size=(k, n))
    best = np.argmin(rank[parents], axis=0)
    return parents[best, np.arange(n)]


def init_population(problem: SWTAProblem, pop_size: int, rng: np.random.Generator,
                    infeasible_ok: bool = False, backend: str = "cpu") -> Population:
    """随机初始化：每个目标每发从可选组合里按 2-锦标赛挑一个（前作 Init.m，最小者胜照抄）。
    infeasible_ok=True（扰动问题上搜索时用）：放弃目标的 [0,0] 占位按发射失败处理，不抛错。
    backend 解码后端（cpu/gpu/auto），交给 evaluate_batch 解析。"""
    N, K = problem.n_target, problem.max_missiles
    decs = np.zeros((pop_size, K, N, 2), dtype=np.int16)
    for t in range(N):
        cand = problem.targets[t].combination
        if cand.shape[0] == 0:
            continue
        picks = tournament_select(cand[:, 2], 2, pop_size * K, rng)
        decs[:, :, t, 0] = cand[picks, 0].reshape(K, pop_size).T
        decs[:, :, t, 1] = cand[picks, 1].reshape(K, pop_size).T
    evs = evaluate_batch(problem, decs, infeasible_ok=infeasible_ok, backend=backend)
    return Population.from_individuals(decs, evs)


def variation(pop: Population, problem: SWTAProblem, rng: np.random.Generator, invert_fitness: bool = False,
              infeasible_ok: bool = False, backend: str = "cpu") -> Population:
    """知识引导交叉 + 单点变异（前作 Operator.m 的移植）。infeasible_ok/backend 同 init_population。"""
    n, K, N = len(pop), problem.max_missiles, problem.n_target
    off = pop.decs.copy()
    for i in range(n):
        p1 = pop.decs[i].copy()
        p2 = pop.decs[rng.integers(n)]
        kdg = pop.kdg[i]

        # 交叉：常出问题的目标位更可能被换掉
        if len(kdg):
            tmp_count = np.bincount(kdg[:, 1], minlength=N).astype(float)
        else:
            tmp_count = np.zeros(N)
        c = (tmp_count + 1) / (tmp_count + 1).sum()
        if rng.random() < 0.5:
            ind = np.nonzero(rng.random(N) < 0.5)[0]
        else:
            ind = rng.choice(N, size=4, replace=True, p=c)
        for pos in ind:
            for k in range(K):
                if rng.random() < 1.0 / K:
                    p1[k, pos] = p2[k, pos]

        # 变异：优先动知识里记过的位置
        mind = np.nonzero(rng.random(N) < 0.1)[0]
        for pos in mind:
            if problem.abandon[pos]:
                continue
            cand = problem.targets[pos].combination
            if len(kdg) and rng.random() < 0.5:
                pick = kdg[rng.integers(len(kdg)), 0]
                fit = -cand[:, 2] if invert_fitness else cand[:, 2]
                pick2 = tournament_select(fit, int(np.ceil(len(cand) / 4)), 1, rng)
                p1[pick, pos] = cand[pick2[0], :2]
            else:
                p1[rng.integers(K), pos] = cand[rng.integers(len(cand)), :2]
        off[i] = p1

    evs = evaluate_batch(problem, off, infeasible_ok=infeasible_ok, backend=backend)
    return Population.from_individuals(off, evs)


def environmental_selection(
    parents: Population, offspring: Population, rng: np.random.Generator, mode: str = "matlab"
) -> Population:
    n = len(parents)
    if mode == "matlab":
        # 前作原码：父代不优于子代（<=）就被同位替换；其后的多样性修剪是死代码，不移植
        replace = parents.objs <= offspring.objs
        new = parents.select(np.arange(n))
        new.decs[replace] = offspring.decs[replace]
        new.objs[replace] = offspring.objs[replace]
        new.kill_prob[replace] = offspring.kill_prob[replace]
        new.fail[replace] = offspring.fail[replace]
        for w in np.nonzero(replace)[0]:
            new.kdg[w] = offspring.kdg[w]
            new.tasks[w] = offspring.tasks[w]
        return new

    if mode == "paper":
        # 论文 Algorithm 3：父子合并，随机抽一个与它最相似的竞争，胜者留下，两者都出列
        objs = np.concatenate([parents.objs, offspring.objs])
        n_kdg = np.array([len(k) for k in parents.kdg] + [len(k) for k in offspring.kdg])
        decs = np.concatenate([parents.decs, offspring.decs])
        flat = decs.reshape(2 * n, -1)
        alive = np.ones(2 * n, dtype=bool)
        keep: list[int] = []
        while len(keep) < n:
            pos = np.nonzero(alive)[0]
            i = int(pos[rng.integers(len(pos))])
            dist = (flat[alive] != flat[i]).sum(axis=1).astype(float)
            dist[list(pos).index(i)] = np.inf
            j = int(pos[int(np.argmin(dist))])
            if objs[i] > objs[j] or (objs[i] == objs[j] and n_kdg[i] <= n_kdg[j]):
                keep.append(i)
            else:
                keep.append(j)
            alive[i] = False
            alive[j] = False
        merged = Population(
            decs=decs,
            objs=objs,
            kdg=list(parents.kdg) + list(offspring.kdg),
            tasks=list(parents.tasks) + list(offspring.tasks),
            kill_prob=np.concatenate([parents.kill_prob, offspring.kill_prob]),
            fail=np.concatenate([parents.fail, offspring.fail]),
        )
        return merged.select(np.array(keep))

    raise ValueError(f"未知环境选择模式：{mode}")


@dataclass
class KCEAResult:
    pop: Population
    trace: np.ndarray  # 每代最优目标值
    elapsed: float
    config: dict = field(default_factory=dict)


def run_kcea(
    problem: SWTAProblem,
    seed: int,
    max_fes: int | None = None,
    pop_size: int | None = None,
    selection: str = "matlab",
    invert_fitness: bool = False,
    infeasible_ok: bool = False,
    backend: str = "cpu",
    verbose: bool = False,
) -> KCEAResult:
    """跑一次 KCEA。max_fes 不传时按论文口径 1000·N；开发调试请显式传 100·N。

    预算口径：每个个体只在名义问题上解码一次，故真实解码量 = 名义 FE。
    最后一代不允许超支（max_fes 非种群整数倍时停在不超过它的最大整数倍；
    100x/1000x/9000x 与 min(2N,500) 种群天然整除，数值轨迹与旧版完全一致）。
    infeasible_ok=True：在缩水后有目标完全打不到的扰动问题上搜索时用
    （放弃目标的占位分配按发射失败处理；名义问题上永远不需要）。
    backend：名义评估后端；"gpu" 走 core.decoder_gpu 批量内核（与逐个体解码逐位一致），
    默认 "cpu" 保持历史路径不变（"auto" 无 GPU 时回退 cpu）。
    """
    selected = resolve_backend(backend)
    rng = np.random.default_rng(seed)
    N = problem.n_target
    max_fes = max_fes or 1000 * N
    pop_size = pop_size or min(2 * N, 500)

    t0 = time.perf_counter()
    pop = init_population(problem, pop_size, rng, infeasible_ok=infeasible_ok, backend=selected)
    fe = pop_size
    trace = [float(pop.objs.max())]
    while fe + len(pop) <= max_fes:
        fit = -pop.objs if invert_fitness else pop.objs
        mating = tournament_select(fit, 2, len(pop), rng)
        offspring = variation(pop.select(mating), problem, rng, invert_fitness, infeasible_ok,
                              backend=selected)
        fe += len(pop)
        pop = environmental_selection(pop, offspring, rng, mode=selection)
        trace.append(float(pop.objs.max()))
        if verbose and len(trace) % 20 == 0:
            print(f"FE={fe:>8} 最优={trace[-1]:.3f}")
    cfg = dict(method="nominal_kcea", seed=seed, max_fes=max_fes, pop_size=pop_size,
               selection=selection, invert_fitness=invert_fitness,
               infeasible_ok=bool(infeasible_ok), backend=selected,
               total_evals=int(fe), actual_decode_evals=int(fe), nominal_fes=int(fe),
               decode_budget=int(max_fes))
    if selected == "gpu":
        cfg["gpu_info"] = gpu_info()
    return KCEAResult(pop, np.array(trace), time.perf_counter() - t0, cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description="跑一次 KCEA 并存下最终种群（多解池）")
    ap.add_argument("--scenario", required=True, help=".mat 场景文件路径")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fes", default="1000x", help="100x / 1000x / 9000x / 具体数字（v3 统一预算用 9000x，旧正式口径 1000x）")
    ap.add_argument("--pop", type=int, default=None, help="种群规模，默认 min(2N,500)")
    ap.add_argument("--selection", default="matlab", choices=["matlab", "paper"])
    ap.add_argument("--invert-fitness", action="store_true", help="锦标赛改回高者胜（非前作行为，只做对照用）")
    ap.add_argument("--load-limit", type=int, default=None, help="覆盖每车弹数上限（默认 15，前作写死；论文 P3/P4 等实例口径不同）")
    ap.add_argument("--n-channel", type=int, default=None, help="覆盖每雷达通道数（默认 10）")
    ap.add_argument("--backend", choices=["cpu", "gpu", "auto"], default="cpu")
    ap.add_argument("--out", required=True, help="输出 .npz 路径")
    args = ap.parse_args()
    selected = resolve_backend(args.backend)

    enemy = load_enemy(args.scenario)
    kw = {}
    if args.load_limit is not None:
        kw["missile_load_limit"] = np.full(6, args.load_limit)
    if args.n_channel is not None:
        kw["radar_n_channel"] = np.full(6, float(args.n_channel))
    params = replace(DefenseParams(), **kw) if kw else None
    problem = SWTAProblem(enemy, params)
    N = problem.n_target
    max_fes = resolve_fes(args.fes, N)

    result = run_kcea(
        problem, seed=args.seed, max_fes=max_fes, pop_size=args.pop,
        selection=args.selection, invert_fitness=args.invert_fitness,
        backend=selected,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        decs=result.pop.decs,
        objs=result.pop.objs,
        trace=result.trace,
        kill_prob=result.pop.kill_prob,
        config=json.dumps(
            {
                **result.config,
                "scenario": str(args.scenario),
                "n_target": N,
                "load_limit": args.load_limit,
                "n_channel": args.n_channel,
            }
        ),
    )
    uniq = len(np.unique(result.pop.decs.reshape(len(result.pop), -1), axis=0))
    print(
        f"场景={Path(args.scenario).name} N={N} 种群={len(result.pop)} "
        f"最优={result.pop.objs.max():.3f} 均值={result.pop.objs.mean():.3f} "
        f"结构不同个体={uniq} 用时={result.elapsed:.1f}s -> {out}"
    )


if __name__ == "__main__":
    main()
