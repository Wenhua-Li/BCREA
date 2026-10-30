# -*- coding: utf-8 -*-
"""GPU 后端验收测试（2026-10-16）：CUDA 融合内核与 CPU 解码逐位一致。

覆盖：
  1. 随机解池（i1 / gen100，正常与扰动问题）：objs/kill_prob/fail/kdg/tasks 全字段逐位一致；
  2. 真实历史解池（data/results/20261013_v3/pools/*.npz）复算逐位一致，并与池内
     objs 存档一致（扫描/分析直接吃这些数，口径不能动）；
  3. infeasible_ok=False 的抛错路径：CPU/GPU 报同一处不可行分配（含打不到目标的占位解）；
  4. run_kcea / run_bcrea 的 CPU 与 GPU 后端轨迹逐位一致（同一随机数流、同一选择）。

GPU 不可用时全部跳过（打印 SKIP）。跑法（仓库根）：
    PYTHONPATH=. py -3.11 tests/test_gpu_backend.py
"""
import json
from pathlib import Path

import numpy as np

import core.robust_metrics as r6
from core.bcrea import BCREAConfig, build_scen_problems, run_bcrea
from experiments.gen_scenarios import _draw_repair
from core import decoder_gpu as dg
from core.decoder import evaluate
from core.kcea import run_kcea
from core.model import DefenseParams, SWTAProblem
from core.scenarios import load_enemy

REPO = Path(__file__).resolve().parents[1]
POOLS = REPO / "data" / "results" / "20261013_v3" / "pools"


def _gpu() -> bool:
    return dg.resolve_backend("auto") == "gpu"


def _rand_pool(prob: SWTAProblem, n: int, seed: int) -> np.ndarray:
    """从每个目标的可选组合里随机采样解池（含打不到目标：留 [0,0] 占位）。"""
    rng = np.random.default_rng(seed)
    decs = np.zeros((n, prob.max_missiles, prob.n_target, 2), dtype=np.int16)
    for i in range(n):
        for t in range(prob.n_target):
            cand = prob.targets[t].combination
            if cand.shape[0] == 0:
                continue
            for k in range(prob.max_missiles):
                decs[i, k, t] = cand[int(rng.integers(cand.shape[0])), :2]
    return decs


def _same_result(c, g) -> bool:
    if c.obj != g.obj or not np.array_equal(c.kill_prob, g.kill_prob):
        return False
    if not np.array_equal(c.fail, g.fail) or not np.array_equal(c.kdg, g.kdg):
        return False
    if len(c.tasks) != len(g.tasks):
        return False
    for a, b in zip(c.tasks, g.tasks):
        if (a.target, a.round_, a.launcher, a.radar, a.row, a.p_kill, a.t_launch, a.t_kill) != \
           (b.target, b.round_, b.launcher, b.radar, b.row, b.p_kill, b.t_launch, b.t_kill):
            return False
    return True


def test_random_pool_bitwise():
    if not _gpu():
        print("SKIP test_random_pool_bitwise（GPU 不可用）")
        return
    cases = [
        ("data/scenarios/tevc/Info_enemy_1.mat", 48, 7),
        ("data/scenarios/generated/gen100_disp_02.mat", 100, 5),
    ]
    for scen, n, seed in cases:
        prob = SWTAProblem(load_enemy(scen), DefenseParams())
        pool = _rand_pool(prob, n, seed)
        a = r6.eval_pool(prob, pool, backend="cpu")
        b = r6.eval_pool(prob, pool, backend="gpu")
        assert a.tobytes() == b.tobytes(), f"{scen} 目标值不一致"
        cpu = [evaluate(prob, pool[i], infeasible_ok=True) for i in range(n)]
        gpu = dg.evaluate_batch(prob, pool, infeasible_ok=True, backend="gpu")
        bad = [i for i in range(n) if not _same_result(cpu[i], gpu[i])]
        assert not bad, f"{scen} 全字段不一致的个体：{bad[:5]}"
    print("test_random_pool_bitwise PASS")


def test_perturbed_pool_bitwise():
    if not _gpu():
        print("SKIP test_perturbed_pool_bitwise（GPU 不可用）")
        return
    prob = SWTAProblem(load_enemy("data/scenarios/tevc/Info_enemy_1.mat"), DefenseParams())
    base = DefenseParams()
    n_r = base.n_radar
    for scale in (0.7, 0.3):   # 0.3 全缩会把目标打成"不可打"，触发占位解路径
        hard = SWTAProblem(prob.enemy, r6.perturbed_params(
            base, np.full(n_r, scale), np.full(r6.n_components(base) - n_r, scale)))
        pool = _rand_pool(hard, 30, 11)
        cpu = [evaluate(hard, pool[i], infeasible_ok=True) for i in range(len(pool))]
        gpu = dg.evaluate_batch(hard, pool, infeasible_ok=True, backend="gpu")
        bad = [i for i in range(len(pool)) if not _same_result(cpu[i], gpu[i])]
        assert not bad, f"scale={scale} 不一致个体：{bad[:5]}"
        msg = []
        for fn in ([lambda: [evaluate(hard, pool[i], infeasible_ok=False) for i in range(len(pool))]],
                   [lambda: dg.evaluate_batch(hard, pool, infeasible_ok=False, backend="gpu")]):
            try:
                fn[0]()
                msg.append(None)
            except ValueError as ex:
                msg.append(str(ex))
        assert msg[0] == msg[1], f"scale={scale} 抛错内容不一致：{msg}"
    print("test_perturbed_pool_bitwise PASS")


def test_real_pools_match_archive():
    if not _gpu():
        print("SKIP test_real_pools_match_archive（GPU 不可用）")
        return
    cands = sorted(POOLS.glob("nominal_i*_s01.npz"))[:2] + sorted(POOLS.glob("bcrea_band_i*_s01.npz"))[:1]
    cands += sorted(POOLS.glob("sampled_wc_i*_s01.npz"))[:1]
    assert cands, f"没有找到历史解池：{POOLS}（换一个已有批次或跳过本测试）"
    for npz in cands:
        d = np.load(npz, allow_pickle=False)
        decs = d["decs"].astype(np.int16)
        objs_stored = d["objs"]
        scen = json.loads(str(d["config"]))["scenario"]
        prob = SWTAProblem(load_enemy(scen), DefenseParams())
        a = r6.eval_pool(prob, decs, backend="cpu")
        b = r6.eval_pool(prob, decs, backend="gpu")
        assert a.tobytes() == b.tobytes(), f"{npz.name} CPU/GPU 不一致"
        assert np.array_equal(a, objs_stored), f"{npz.name} 与池内 objs 存档不一致"
    print(f"test_real_pools_match_archive PASS（{len(cands)} 个池）")


def test_kcea_backend_trace_identical():
    if not _gpu():
        print("SKIP test_kcea_backend_trace_identical（GPU 不可用）")
        return
    enemy, _, bad = _draw_repair(12, "tight", 8101)
    assert len(bad) == 0
    prob = SWTAProblem(enemy, DefenseParams())
    r1 = run_kcea(prob, seed=1, max_fes=96)
    r2 = run_kcea(prob, seed=1, max_fes=96, backend="gpu")
    assert np.array_equal(r1.trace, r2.trace), (r1.trace, r2.trace)
    assert np.array_equal(r1.pop.objs, r2.pop.objs)
    assert np.array_equal(r1.pop.decs, r2.pop.decs)
    assert r2.config["backend"] == "gpu" and r1.config["backend"] == "cpu"
    print("test_kcea_backend_trace_identical PASS")


def test_bcrea_backend_trace_identical():
    if not _gpu():
        print("SKIP test_bcrea_backend_trace_identical（GPU 不可用）")
        return
    enemy, _, bad = _draw_repair(12, "tight", 8101)
    assert len(bad) == 0
    prob = SWTAProblem(enemy, DefenseParams())
    cfg = BCREAConfig(n_scen=4, scen_seed=9001)
    scen = build_scen_problems(prob.enemy, DefenseParams(), cfg)
    p1, d1, t1, td1, _, c1 = run_bcrea(prob, scen, seed=1, max_fes=50, cfg=cfg)
    p2, d2, t2, td2, _, c2 = run_bcrea(prob, scen, seed=1, max_fes=50, cfg=cfg, backend="gpu")
    assert np.array_equal(t1, t2) and np.array_equal(td1, td2), (t1, t2)
    assert np.array_equal(p1.objs, p2.objs) and np.array_equal(p1.decs, p2.decs)
    assert c2["backend"] == "gpu"
    print("test_bcrea_backend_trace_identical PASS")


if __name__ == "__main__":
    test_random_pool_bitwise()
    test_perturbed_pool_bitwise()
    test_real_pools_match_archive()
    test_kcea_backend_trace_identical()
    test_bcrea_backend_trace_identical()
    print("test_gpu_backend: 全部 PASS")
