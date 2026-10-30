"""鲁棒度量工具箱：预算式扰动集合 + 扰动重评估 + 鲁棒度量（依赖 core 模型与解码器）。

不确定模型（选题档案 7.1 设计红线；红线#2 表述按用户 2026-10-02 拍板口径）：
  - 只扰"会改变可行性"的参数：6 部雷达的探测威力 + 6 部发射车的杀伤边界，共 12 个分量；
    每个分量要么保持名义、要么缩水 δ（范围 ×(1-δ)）。
    不直接扰 pKill 等概率值——目标函数对它们单调递增，一起缩只会整体平移（角点塌缩）。
  - 预算式（Γ）集合：每个扰动场景恰好选 Γ 个分量缩水（Γ≥分量数时=全缩，即"同一角点"对照）。
  - 价值定位（拍板口径）：最坏退化作为鲁棒度量 + 多样且稳健的解集；
    不强调"预算式 Γ 逼出不同解怕不同扰动"（试跑实测大 δ 下存在通用最坏场景）。

度量（在名义等价集 E 上：obj0 >= best*(1-band_rel)，即"指挥员眼中差不多一样好"）：
  - 单解最坏退化 Δ(x) = obj0(x) − min_s obj_s(x)（terminology.md 已登记）；
  - spread_wc = E 内最坏目标极差；amp = spread_wc / E 内名义极差（放大倍数）；
  - 解集鲁棒覆盖 RC(X) = min_s max_{x∈X} obj_s(x)：不管来哪种扰动，集合里最好方案至少顶到这个水平。

扰动场景采样对同一 (场景文件, δ, Γ, scen_seed) 确定可复现——跨解池种子共享同一批场景，保证可比。
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from core.decoder import evaluate
from core.decoder_gpu import decode_objs, gpu_info, resolve_backend  # noqa: F401（CLI/测试从这里取）

from core.model import DefenseParams, SWTAProblem
from core.scenarios import EnemyInfo


def nominal_unit_ranges(params: DefenseParams) -> np.ndarray:
    """按"部"的名义探测威力 (nR,3)：把按类型的表展开到每部（对齐 MATLAB Radar.R(j,:) 口径）。"""
    return np.asarray(params.radar_R)[np.asarray(params.radar_type) - 1]


def n_components(params: DefenseParams) -> int:
    """不确定分量个数 = 雷达部数 + 发射车数（默认部署 6+6=12）。"""
    return params.n_radar + params.n_missile


def stratified_masks(k: int, gamma: int, seed: int, n_comp: int) -> np.ndarray:
    """分层评分场景（k 个）：先把全部 n_comp 个分量按 gamma 分组铺满（保证每个分量至少被一个
    训练场景覆盖——修 seed 敏感性：随机 K 场景可能漏掉关键分量，rkss9003 退化 3.6 vs 9001 的 0.17），
    剩余名额随机补。k < ceil(n_comp/gamma) 时退化为覆盖优先的前 k 组。"""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_comp)
    masks = []
    for i in range(0, n_comp, gamma):
        chunk = perm[i:i + gamma]
        if len(masks) >= k:
            break
        m = np.zeros(n_comp, dtype=np.int8)
        m[chunk] = 1
        masks.append(m)
    while len(masks) < k:
        m = np.zeros(n_comp, dtype=np.int8)
        m[rng.choice(n_comp, size=gamma, replace=False)] = 1
        masks.append(m)
    return np.array(masks[:k])


def cell_seed(scen_seed: int, delta: float, gamma: int) -> int:
    """(δ,Γ) 单元的场景采样种子：同一单元跨解池种子共享同一批扰动场景。"""
    return scen_seed + 100000 * int(round(delta * 100)) + 17 * int(gamma)


def sample_masks(gamma: int, n_scen: int, seed: int, n_comp: int, enum_cap: int = 250) -> np.ndarray:
    """预算式扰动掩码 (m, n_comp) 0/1：每行恰好 gamma 个分量缩水。

    gamma>=n_comp：全缩（"同一角点"对照），唯一场景，只返回 1 行。
    C(n_comp,gamma) <= enum_cap：**确定性穷举**全部组合（最坏情形精确、跨任务天然一致，
      12 分量时 Γ=1/2/3 分别 12/66/220 个，成本与采样相当）；否则随机采样 n_scen 个。
    enum_cap=0 强制采样（冒烟提速用）。"""
    if gamma >= n_comp:
        return np.ones((1, n_comp), dtype=np.int8)
    from itertools import combinations
    from math import comb

    total = comb(n_comp, gamma)
    if total <= enum_cap:
        masks = np.zeros((total, n_comp), dtype=np.int8)
        for r, idx in enumerate(combinations(range(n_comp), gamma)):
            masks[r, list(idx)] = 1
        return masks
    rng = np.random.default_rng(seed)
    masks = np.zeros((n_scen, n_comp), dtype=np.int8)
    for s in range(n_scen):
        masks[s, rng.choice(n_comp, size=gamma, replace=False)] = 1
    return masks


def scales_from_mask(mask: np.ndarray, delta: float, n_radar: int) -> tuple[np.ndarray, np.ndarray]:
    """掩码行 → (雷达缩放 (nR,), 发射车缩放 (nW,))，缩水分量 ×(1-δ)。"""
    mask = np.asarray(mask, dtype=float)
    return 1.0 - delta * mask[:n_radar], 1.0 - delta * mask[n_radar:]


def perturbed_params(base: DefenseParams, rad_scale: np.ndarray, mis_scale: np.ndarray) -> DefenseParams:
    """按缩放因子生成扰动参数：只走扰动钩子字段（radar_R_unit/rkill_scale_unit），不动公共字段。"""
    return replace(
        base,
        radar_R_unit=nominal_unit_ranges(base) * np.asarray(rad_scale, dtype=float)[:, None],
        rkill_scale_unit=np.asarray(mis_scale, dtype=float),
    )


def eval_pool(problem: SWTAProblem, decs_pool: np.ndarray, backend: str = "cpu", batch_size: int = 256) -> np.ndarray:
    """批量评估解池；默认 CPU 以保持历史调用和结果口径不变。

    gpu 走 core.decoder_gpu 的单融合 kernel（与逐个体 evaluate 逐位一致，
    见 tests/test_gpu_backend.py）；batch_size 只作为分块上限，不改变结果。
    """
    selected = resolve_backend(backend)
    if selected == "gpu":
        return decode_objs(problem, np.asarray(decs_pool), batch_size=batch_size)
    return np.array([evaluate(problem, decs_pool[i], infeasible_ok=True).obj for i in range(len(decs_pool))])


def equiv_set(objs0: np.ndarray, band_rel: float = 0.005) -> np.ndarray:
    """名义等价集下标：obj0 >= best*(1-band_rel)；成员<2 时退回严格最优集。"""
    best = float(objs0.max())
    mask = objs0 >= best * (1.0 - band_rel)
    if mask.sum() < 2:
        mask = objs0 >= best - 1e-9
    return np.nonzero(mask)[0]


def n_struct_distinct(decs_pool: np.ndarray, idx: np.ndarray) -> int:
    """idx 指定子集里结构互异的解数（decs 全拉平比较）。"""
    flat = decs_pool[idx].reshape(len(idx), -1)
    return int(np.unique(flat, axis=0).shape[0])


def cell_metrics(objs0: np.ndarray, objsS: np.ndarray, eq_idx: np.ndarray) -> dict:
    """一个 (δ,Γ) 单元的扰动度量。objsS: (n_scen, n_pool)。"""
    objs_wc = objsS.min(axis=0)
    argmin = objsS.argmin(axis=0)
    deg = objs0 - objs_wc
    e = eq_idx
    spread_obj0 = float(objs0[e].max() - objs0[e].min())
    spread_wc = float(objs_wc[e].max() - objs_wc[e].min())
    spread_deg = float(deg[e].max() - deg[e].min())
    best = float(objs0.max())
    return {
        "objs_wc": objs_wc,
        "argmin": argmin,
        "deg": deg,
        "spread_wc": spread_wc,
        "spread_deg": spread_deg,
        "amp": spread_wc / spread_obj0 if spread_obj0 > 1e-12 else float("inf"),
        "rel_spread": spread_deg / best if best > 0 else float("nan"),
        "reorders": int(len(np.unique(np.round(objs_wc[e], 6))) > 1),
        "n_distinct_worst": int(len(np.unique(argmin[e]))),
    }


def robust_coverage(objsS: np.ndarray, members: np.ndarray) -> float:
    """解集鲁棒覆盖 RC(X) = min_s max_{x∈X} obj_s(x)。objsS: (n_scen, n_pool)。"""
    return float(objsS[:, np.asarray(members)].max(axis=1).min())


def greedy_rc_select(objsS: np.ndarray, m: int, tie_break_decs: np.ndarray | None = None) -> list[int]:
    """贪心选 m 个解最大化 RC：每步加入"最能抬高 min_s max_x"的解。
    （主结论实验的'集合最优'策略基准；对比 top-m 名义 / top-m 鲁棒 / KCEA 原池。）

    tie_break_decs（可选，(n_pool, ...) 决策矩阵）：并列时按结构距离破平——
    RC 增益差 <1e-9 视为并列，并列候选里挑"与已选集合最小 Hamming 距离最大"者；
    尚无已选时取第一个最高 RC（规则与 analyze_deep.py 的双准则贪心一致，2026-10-13 迁入）。
    默认 None = 纯贪心，与历史行为逐点一致（argmax 取首个最大值）。
    """
    n = objsS.shape[1]
    chosen: list[int] = []
    cur_max = np.full(objsS.shape[0], -np.inf)
    flat = tie_break_decs.reshape(n, -1) if tie_break_decs is not None else None
    for _ in range(min(m, n)):
        new_max = np.maximum(cur_max[None, :], objsS.T)  # (n_pool, n_scen)
        rc = new_max.min(axis=1)
        rc[chosen] = -np.inf
        if flat is None:
            i = int(rc.argmax())
        else:
            best_val = rc.max()
            tied = np.nonzero(rc >= best_val - 1e-9)[0]
            if len(tied) > 1 and chosen:
                dmin = np.array([(flat[t] != flat[chosen]).sum(axis=1).min() for t in tied])
                i = int(tied[dmin.argmax()])
            else:
                i = int(tied[0])
        chosen.append(i)
        cur_max = np.maximum(cur_max, objsS[:, i])
    return chosen


def extract_menu(scan: dict, decs_pool: np.ndarray, m: int = 5,
                 delta: float = 0.3, gamma: int = 2, tie_break: bool = True) -> dict:
    """从 exp_robust 扫描 JSON（已 load 成 dict）+ 池 npz 的 decs 提取固定认证菜单。

    用法示例：
        js = json.load(open("扫描.json")); decs = np.load("池.npz")["decs"]
        menu = extract_menu(js, decs, m=5)        # menu["members"] 是池内全局下标
    步骤：取指定 (δ,Γ) 单元的 objsS → 名义等价集 eq 上跑 greedy_rc_select
    （tie_break=True 时用 decs 做结构距离破平）→ 返回成员下标与该菜单的 RC。
    """
    cell = scan["cells"][f"d{round(delta * 100):02d}_g{gamma}"]
    objsS = np.asarray(cell["objsS"], dtype=float)
    eq = np.asarray(scan["nominal"]["eq_idx"], dtype=int)
    tb = np.asarray(decs_pool)[eq] if tie_break else None
    sel_local = greedy_rc_select(objsS[:, eq], m, tie_break_decs=tb)
    members = eq[np.asarray(sel_local, dtype=int)]
    return {
        "members": members.tolist(),       # 池内全局下标（可直接索引 decs/objs0）
        "m": int(m), "delta": float(delta), "gamma": int(gamma),
        "tie_break": bool(tie_break),
        "rc": robust_coverage(objsS, members),
    }


def scan_cell(
    enemy: EnemyInfo,
    base_params: DefenseParams,
    decs_pool: np.ndarray,
    delta: float,
    gamma: int,
    n_scen: int,
    scen_seed: int,
    enum_cap: int = 250,
    backend: str = "cpu",
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, float]:
    """跑一个 (δ,Γ) 单元：生成场景（可穷举）→ 逐场景建扰动问题并评估解池。
    返回 (masks, objsS (n_scen,n_pool), 耗时秒)。"""
    import time

    n_radar = base_params.n_radar
    n_comp = n_components(base_params)
    masks = sample_masks(gamma, n_scen, cell_seed(scen_seed, delta, gamma), n_comp, enum_cap)
    t0 = time.perf_counter()
    rows = []
    for mask in masks:
        rad, mis = scales_from_mask(mask, delta, n_radar)
        prob = SWTAProblem(enemy, perturbed_params(base_params, rad, mis))
        rows.append(eval_pool(prob, decs_pool, backend=backend, batch_size=batch_size))
    return masks, np.array(rows), time.perf_counter() - t0
