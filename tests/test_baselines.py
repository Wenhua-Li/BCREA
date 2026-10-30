# -*- coding: utf-8 -*-
"""基线单元测试（评审 GPT-B3 的两项复现检查，修复后必须通过）。

跑法（仓库根）：PYTHONPATH=. python tests/test_baselines.py
"""
import numpy as np

from experiments.run_nsga2_pool import crowding, pareto_fronts


def test_pareto_ties_same_front():
    """三个完全相同的目标向量：非支配并列 → 同属第一层（旧 bug：[[0],[1],[2]]）。"""
    f = np.array([0.0, 0.0, 0.0])
    fronts = pareto_fronts(f, f)
    assert sorted(fronts[0].tolist()) == [0, 1, 2], f"并列被拆层: {fronts}"


def test_pareto_dominated_split():
    f1 = np.array([0.0, 1.0, 2.0])
    f2 = np.array([0.0, 1.0, 2.0])
    fronts = pareto_fronts(f1, f2)
    assert [fr.tolist() for fr in fronts] == [[0], [1], [2]], fronts


def test_crowding_inf_no_nan_in_tournament():
    """两点前沿的拥挤距离=inf：字典序锦标赛不得产生 NaN（旧 bug：inf/inf=NaN）。"""
    cd = crowding(np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0, 1]))
    assert np.isinf(cd).all() or np.isfinite(cd).all()
    rank = np.zeros(2, dtype=int)
    a, b = np.array([0, 1]), np.array([1, 0])
    a_wins = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (cd[a] > cd[b]))
    assert a_wins.dtype == bool and not np.isnan(cd).any()


if __name__ == "__main__":
    test_pareto_ties_same_front()
    test_pareto_dominated_split()
    test_crowding_inf_no_nan_in_tournament()
    print("test_baselines: 3/3 PASS")
