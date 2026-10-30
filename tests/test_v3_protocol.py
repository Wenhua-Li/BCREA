# -*- coding: utf-8 -*-
"""v3 协议单元测试（2026-10-13，"锁配置"阶段）。

覆盖：
  1. 预算不超支：bcrea / nsga2 / sampled_wc 给了 --decode-budget 后 actual_decode_evals ≤ 预算；
     kcea 非种群整数倍预算不超支，整数倍（100x/1000x 口径）轨迹不变。
  2. band/strict 标签：no_downgrade=False→bcrea_band，True→bcrea_strict（不再用含糊的 bcrea_v0）。
  3. 训练掩码存档：config 里 train_masks 与 stratified_masks 现算一致。
  4. greedy_rc_select：默认（不传 decs）与旧纯贪心逐点一致；传 decs 时 RC 并列按结构距离破平。
  5. 不可行扰动搜索：新实例在悲观/角点退化下允许目标完全打不到；[0,0] 占位按失败处理，
     与 exp_robust 的扰动评估语义一致。
  6. 任务清单数量：主池 30×6×10=1800、扫描每池一条=1800、reopt 基础清单 6×66×10=3960
     （生成器口径；正式批次按 2026-10-15 拍板用 5 种子 = 1980 条，见
     tasks_20261013_v3_reopt_5seed.txt）；预算 D=9000·N。

跑法（仓库根）：PYTHONPATH=. python tests/test_v3_protocol.py
"""
import json
from itertools import combinations

import numpy as np

import core.robust_metrics as r6
from core.bcrea import BCREAConfig, build_scen_problems, run_bcrea, selection_label
from experiments.run_sampled_wc import run_sampled_wc
from experiments.gen_scenarios import _draw_repair
from experiments.run_nsga2_pool import run_nsga2
from core.kcea import resolve_fes, run_kcea
from core.model import DefenseParams, SWTAProblem

POP = 24          # 小实例 N=12 → pop=min(2N,500)=24
BUDGET = 290      # 非整数倍：初始 24*(1+4)=120，每代 120；只够 1 代（240），第 2 代 360 超支


def _tiny_problem():
    """确定可复现的 12 目标小场景（gen_scenarios 同一采样器，失败即报错换不了种子）。"""
    enemy, rounds, bad = _draw_repair(12, "tight", 8101)
    assert len(bad) == 0, f"小场景重试 {rounds} 轮后仍有打不到的目标：{bad.tolist()}"
    return SWTAProblem(enemy, DefenseParams())


def _scen(nominal, n_scen=4, scen_seed=9001):
    cfg = BCREAConfig(n_scen=n_scen, scen_seed=scen_seed)
    probs = build_scen_problems(nominal.enemy, DefenseParams(), cfg)
    return cfg, probs


def test_eval_pool_backend_fallback_and_cpu_reference():
    """GPU 不可用时 auto 回退；可用时 GPU 与 CPU 小池逐项一致。"""
    nominal = _tiny_problem()
    rng = np.random.default_rng(17)
    decs = np.zeros((4, nominal.max_missiles, nominal.n_target, 2), dtype=np.int16)
    for i in range(len(decs)):
        for t in range(nominal.n_target):
            cand = nominal.targets[t].combination
            if len(cand):
                decs[i, :, t] = cand[rng.integers(len(cand)), :2]
    cpu = r6.eval_pool(nominal, decs)
    assert r6.resolve_backend("auto") in {"cpu", "gpu"}
    if r6.gpu_info()["available"]:
        gpu = r6.eval_pool(nominal, decs, backend="gpu", batch_size=2)
        assert np.allclose(cpu, gpu, rtol=0, atol=1e-10), (cpu, gpu)
    else:
        assert np.array_equal(cpu, r6.eval_pool(nominal, decs, backend="auto"))


def test_budget_no_overshoot():
    nominal = _tiny_problem()
    cfg, scen = _scen(nominal)

    # BCREA：预算模式不超支；legacy 模式保持 --fes 行为但记账齐全
    _, _, _, _, _, cd = run_bcrea(nominal, scen, seed=1, max_fes=10**9, cfg=cfg,
                                  decode_budget=BUDGET)
    assert cd["actual_decode_evals"] <= BUDGET, cd
    assert cd["actual_decode_evals"] == 240 and cd["nominal_fes"] == 48, cd
    assert cd["decode_budget"] == BUDGET and cd["decode_budget_remaining"] == BUDGET - 240, cd
    _, _, _, _, _, cd0 = run_bcrea(nominal, scen, seed=1, max_fes=100, cfg=cfg)
    assert cd0["nominal_fes"] == 120, cd0          # 旧行为：fe 到 120 才越线停
    assert cd0["decode_budget"] == 100 * 5 and cd0["actual_decode_evals"] == 600, cd0

    # NSGA-II：预算模式不超支，记账齐全
    _, _, _, _, cm = run_nsga2(nominal, scen, seed=1, max_fes=10**9, cfg=cfg,
                               decode_budget=BUDGET)
    assert cm["actual_decode_evals"] <= BUDGET and cm["actual_decode_evals"] == 240, cm
    assert cm["method"] == "nsga2" and cm["stratified"] and cm["nominal_fes"] == 48, cm

    # sampled_wc：预算含结尾名义复核预留，总量仍不超支
    cfg2, scen2 = _scen(nominal, n_scen=4, scen_seed=9101)
    _, _, _, _, _, cw = run_sampled_wc(nominal, scen2, seed=1, max_fes=10**9, cfg=cfg2,
                                       decode_budget=BUDGET)
    assert cw["actual_decode_evals"] <= BUDGET, cw
    assert cw["method"] == "sampled_worst_case" and cw["stratified"], cw
    assert cw["actual_decode_evals"] == cw["total_evals"], cw

    # KCEA：非整数倍预算不超支；整数倍（等价 100x/1000x 口径）轨迹不变
    res = run_kcea(nominal, seed=1, max_fes=100)
    assert res.config["method"] == "nominal_kcea" and res.config["total_evals"] == 96
    assert res.config["actual_decode_evals"] == 96, res.config
    assert res.config["nominal_fes"] == 96 and res.config["decode_budget"] == 100, res.config
    assert len(res.trace) == 1 + 3                  # 初始 + 3 代（96/24）
    res2 = run_kcea(nominal, seed=1, max_fes=96)
    assert res2.config["total_evals"] == 96 and len(res2.trace) == len(res.trace)
    assert np.array_equal(res.trace, res2.trace)    # 整除边界处轨迹一致
    print("test_budget_no_overshoot PASS")


def test_band_strict_labels():
    assert selection_label(BCREAConfig(no_downgrade=False)) == "bcrea_band"
    assert selection_label(BCREAConfig(no_downgrade=True)) == "bcrea_strict"
    nominal = _tiny_problem()
    cfg, scen = _scen(nominal)
    _, _, _, _, _, cb = run_bcrea(nominal, scen, seed=1, max_fes=50, cfg=cfg)
    assert cb["method"] == "bcrea_band" and cb["selection"] == "bcrea_band" and cb["no_downgrade"] is False, cb
    cfg_s = BCREAConfig(n_scen=4, scen_seed=9001, no_downgrade=True)
    _, _, _, _, _, cs = run_bcrea(nominal, scen, seed=1, max_fes=50, cfg=cfg_s)
    assert cs["method"] == "bcrea_strict" and cs["selection"] == "bcrea_strict" and cs["no_downgrade"] is True, cs
    print("test_band_strict_labels PASS")


def test_train_masks_archived():
    nominal = _tiny_problem()
    cfg, scen = _scen(nominal)
    _, _, _, _, _, cd = run_bcrea(nominal, scen, seed=1, max_fes=50, cfg=cfg)
    want = r6.stratified_masks(4, 2, 9001, r6.n_components(nominal.params)).astype(int).tolist()
    assert cd["train_masks"] == want, "train_masks 与分层采样现算不一致"
    _, _, _, _, cm = run_nsga2(nominal, scen, seed=1, max_fes=50, cfg=cfg)
    assert cm["train_masks"] == want, "nsga2 的 train_masks 与 bcrea 不一致"
    cfg2, scen2 = _scen(nominal, n_scen=4, scen_seed=9101)
    _, _, _, _, _, cw = run_sampled_wc(nominal, scen2, seed=1, max_fes=50, cfg=cfg2)
    want2 = r6.stratified_masks(4, 2, 9101, 12).astype(int).tolist()
    assert cw["train_masks"] == want2
    print("test_train_masks_archived PASS")


def _old_greedy(objsS: np.ndarray, m: int) -> list[int]:
    """greedy_rc_select 的旧实现（纯贪心，逐字留档做回归对照）。"""
    n = objsS.shape[1]
    chosen: list[int] = []
    cur_max = np.full(objsS.shape[0], -np.inf)
    for _ in range(min(m, n)):
        new_max = np.maximum(cur_max[None, :], objsS.T)
        rc = new_max.min(axis=1)
        rc[chosen] = -np.inf
        i = int(rc.argmax())
        chosen.append(i)
        cur_max = np.maximum(cur_max, objsS[:, i])
    return chosen


def test_greedy_tie_break():
    rng = np.random.default_rng(42)
    for _ in range(30):
        S = rng.random((6, 9))
        m = int(rng.integers(1, 6))
        assert r6.greedy_rc_select(S, m) == _old_greedy(S, m), "默认行为与旧纯贪心不一致"

    # 并列破平：候选 1/2/3 的 RC 增益完全并列，decs 让 2 号离已选最远
    S = np.array([[5.0, 4.0, 4.0, 1.0], [5.0, 4.0, 4.0, 1.0]])
    decs = np.array([[0, 0], [0, 0], [1, 1], [0, 1]], dtype=np.int16)
    plain = r6.greedy_rc_select(S, 2)
    assert plain == [0, 1], plain                    # 默认取首个最大值（旧行为）
    tb = r6.greedy_rc_select(S, 2, tie_break_decs=decs)
    assert tb == [0, 2], tb                          # 并列里挑结构最远的 2 号
    assert r6.robust_coverage(S, plain) == r6.robust_coverage(S, tb), "破平不许牺牲 RC"

    # extract_menu 小函数：扫描 JSON + 池 decs → 固定菜单
    scan = {"nominal": {"eq_idx": [0, 1, 2, 3]},
            "cells": {"d30_g2": {"objsS": S.tolist()}}}
    menu = r6.extract_menu(scan, decs, m=2)
    assert menu["members"] == [0, 2] and abs(menu["rc"] - 5.0) < 1e-9, menu
    print("test_greedy_tie_break PASS")


def test_infeasible_perturbed_search():
    """扰动后有目标完全打不到时，搜索按发射失败继续；默认口径仍视为上游错误。"""
    nominal = _tiny_problem()
    base = DefenseParams()
    n_r, n_c = base.n_radar, r6.n_components(base)
    pess = SWTAProblem(nominal.enemy, r6.perturbed_params(base, np.full(n_r, 0.7),
                                                          np.full(n_c - n_r, 0.7)))
    assert int(pess.abandon.sum()) == 1
    res = run_kcea(pess, seed=1, max_fes=96, infeasible_ok=True)
    assert res.config["infeasible_ok"] is True
    assert np.all(np.isfinite(res.pop.objs)) and res.pop.objs.max() <= pess.n_target - 1
    try:
        run_kcea(pess, seed=1, max_fes=24)
    except ValueError:
        pass
    else:
        raise AssertionError("infeasible_ok=False 时应对占位不可行分配报错")
    print("test_infeasible_perturbed_search PASS")


def test_kcea_fes_mapping():
    assert resolve_fes("100x", 24) == 2400
    assert resolve_fes("1000x", 24) == 24000
    assert resolve_fes("9000x", 24) == 216000
    assert resolve_fes("9000x", 100) == 900000
    assert resolve_fes("1234", 24) == 1234
    print("test_kcea_fes_mapping PASS")


def test_task_lists():
    from experiments import make_tasks as mt

    assert len(mt.INSTANCES) == 30, len(mt.INSTANCES)
    assert mt.budget_of("i1") == 216000 and mt.budget_of("i11") == 432000
    assert mt.budget_of("i100") == 900000 and mt.budget_of("gen24_conc_01") == 216000

    pools = mt.pool_tasks()
    scans = mt.scan_tasks()
    reopt = mt.reopt_tasks()
    assert len(pools) == 1800, len(pools)
    assert len(scans) == 1800, len(scans)
    assert len(reopt) == 6 * len(list(combinations(range(12), 2))) * 10 == 3960, len(reopt)

    # 抽查命令行口径：预算、标签、场景路径
    band = [c for c, m in pools if m["method"] == "bcrea_band" and m["instance"] == "i1"
            and m["seed"] == 1][0]
    assert "--decode-budget 216000" in band and "--scen-seed 9001" in band
    assert "--no-downgrade" not in band and "data/scenarios/tevc/Info_enemy_1.mat" in band
    strict = [c for c, m in pools if m["method"] == "bcrea_strict" and m["instance"] == "i1"
              and m["seed"] == 1][0]
    assert "--no-downgrade" in strict and band.replace("_band_", "_strict_") != strict
    nom = [c for c, m in pools if m["method"] == "nominal" and m["instance"] == "gen100_disp_02"][0]
    assert "--fes 9000x" in nom and "data/scenarios/generated/gen100_disp_02.mat" in nom
    wc = [c for c, m in pools if m["method"] == "sampled_wc" and m["instance"] == "i11"][0]
    assert "--m-scen 16" in wc and "--scen-seed 9101" in wc and "--decode-budget 432000" in wc
    re0 = [c for c, m in reopt if m["instance"] == "gen24_conc_01" and m["seed"] == 1][0]
    assert "run_corner_pool.py" in re0 and "--fes 9000x" in re0

    # 实际落盘一次（本任务的要求：只生成清单，不启动任务）
    mt.main()
    manifest = json.loads((mt.TASKS_DIR / "tasks_20261013_v3_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 1800 + 1800 + 3960, len(manifest)
    print("test_task_lists PASS")


if __name__ == "__main__":
    test_kcea_fes_mapping()
    test_infeasible_perturbed_search()
    test_greedy_tie_break()
    test_task_lists()
    test_band_strict_labels()
    test_train_masks_archived()
    test_budget_no_overshoot()
    print("test_v3_protocol: 7/7 PASS")
