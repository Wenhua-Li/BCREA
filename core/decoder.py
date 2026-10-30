"""SWTA 解码器（前作 CalObj.m 的移植）：把分配方案解码成任务序列并算目标值。

方案表示：decs[k, t, 0]=第 k 发由哪部发射车打目标 t，decs[k, t, 1]=制导雷达。
下标一律 0 基。解码顺序 = problem.target_ind（飞行时间短的目标先处理）。

与 MATLAB 原码的两处有意偏差（2026-10-02 移植时核定，其余全部照抄）：
  1. 通道冲突调整：原码 nM==1 分支里 while 循环推进发射时刻但时间窗下标不更新，
     重查条件永远不变，等价于"冲突即失败"。这里直写等价语义。
  2. 行下标越界防护：概率表尾部全零行在建表时已截断，走到表尾即视为不可行，
     原码 100 目标崩溃（Operator.m:14 一带）在这条路径上不再有越界机会。

另注（2026-10-16 GPU 后端移植时核定）：
  通道占用检查里 occupied 的切片 `occupied[max(a, s0) - 1 : min(b, s1)]` 用的是全局槽
  下标去切长度 s1-s0+1 的局部数组，只有 s0 == 1 时才对位；s0 > 1 时起点落在数组界外
  （Python 切片截断为"不标记"）。这是既有全部实验结果的真实口径，GPU 后端
  （core/decoder_gpu.py）已按同义复刻。不要"顺手修正"——改成正确下标会改变历史
  数字，必须连全部实验一起重跑，且与本文件已定格的 660 个池产物不再可比。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.model import SWTAProblem


@dataclass
class Task:
    """一次成功的发射任务（解码产物，解释生成器直接用）。"""

    target: int  # 目标下标（0 基）
    round_: int  # 第几发（0 基）
    launcher: int  # 发射车下标
    radar: int  # 制导雷达下标
    row: int  # 用了概率表第几行（0 基；行越靠后命中概率越低）
    p_kill: float
    t_launch: float
    t_kill: float


@dataclass
class EvalResult:
    """一个方案（个体）的解码结果。"""

    obj: float  # 期望毁伤目标数（越大越好）
    kdg: np.ndarray  # (m,2) 知识行 [第几发(0基), 处理顺序位置(0基)]，可重复
    tasks: list[Task] = field(default_factory=list)
    kill_prob: np.ndarray | None = None  # (N, K+1)：每发命中概率 + 末列目标总命中概率
    fail: np.ndarray | None = None  # (N, K) 哪几发没发出去


def _matlab_round(x: np.ndarray | float) -> np.ndarray | float:
    # MATLAB round 是"远离零取半"，numpy 是"取偶"；正数场景下用 floor(x+0.5) 对齐
    return np.floor(np.asarray(x) + 0.5)


def evaluate(problem: SWTAProblem, decs: np.ndarray, infeasible_ok: bool = False) -> EvalResult:
    """解码一个方案。decs 形状 (K, N, 2)。只访问该目标可选组合里的（发射车, 雷达）对。

    infeasible_ok=False（默认，行为不变）：decs 出现不可行对视为上游 bug，抛 ValueError。
    infeasible_ok=True（扰动评估用）：参数缩水后"原可行的对变不可行"是预期行为，
    按"该目标第 k..K 发全部失败"处理（与 MATLAB 原码失败分支、扰动评估修复版 CalObj 一致）。
    """
    p = problem.params
    N, K = problem.n_target, problem.max_missiles
    order = problem.target_ind
    d1lo, d1hi = p.delta_t1
    cap = p.radar_n_channel[p.radar_type - 1]  # 每部雷达的通道上限（按其类型）
    need_tbl = p.radar_coefficient
    rtype = p.radar_type - 1
    load_limit = p.missile_load_limit
    mtype = p.missile_type - 1

    launcher_used = np.zeros(p.n_missile, dtype=np.int32)
    colsum = np.zeros((p.n_radar, problem.nS), dtype=np.int64)  # 每雷达每时间槽已占通道
    occ: dict[tuple[int, int], list[tuple[int, int]]] = {}  # (雷达,目标)->已占时间槽区间（1 基闭区间）

    kill_prob = np.zeros((N, K))
    fail = np.zeros((N, K), dtype=bool)
    last_kill_t = np.full(N, np.nan)
    last_radar = np.full(N, -1, dtype=int)
    tasks: list[Task] = []
    kdg: list[tuple[int, int]] = []

    for k in range(K):
        for pos in range(N):
            t = int(order[pos])
            if fail[t, k]:
                continue
            launcher = int(decs[k, t, 0])
            radar = int(decs[k, t, 1])
            if launcher_used[launcher] >= load_limit[mtype[launcher]]:
                fail[t, k:] = True
                continue
            tbl = problem.targets[t].pkill.get((launcher, radar))
            if tbl is None or len(tbl) == 0:
                if infeasible_ok:
                    fail[t, k:] = True
                    continue
                # decs 里出现了该目标可选组合之外的对：构造方案的上游出了问题
                raise ValueError(f"目标{t}第{k}发分配了不可行的 (发射车{launcher}, 雷达{radar})")

            row = 0
            if k > 0:
                t_kill = tbl[row, 2]
                delta = t_kill - last_kill_t[t]
                ok = d1lo <= delta <= d1hi
                t_cap = problem.targets[t].T  # 前作拿飞行秒数当行数上限用，照抄
                while not ok:
                    row += 1
                    if row >= len(tbl) or row > t_cap:
                        kdg.append((k, pos))
                        break
                    t_kill = tbl[row, 2]
                    delta = t_kill - last_kill_t[t]
                    ok = d1lo <= delta <= d1hi
                if delta < p.preT and last_radar[t] != radar:
                    kdg.append((k, pos))
                    ok = False
                if not ok:
                    fail[t, k:] = True
                    kdg.append((k, pos))
                    continue

            t_kill = tbl[row, 2]
            s0 = max(int(_matlab_round((t_kill - p.preT) / p.dT)), 1)
            s1 = min(int(_matlab_round(t_kill / p.dT)), problem.nS)
            need = int(need_tbl[rtype[radar], problem.ttype[t] - 1])
            base = colsum[radar, s0 - 1 : s1]
            occupied = np.zeros(s1 - s0 + 1, dtype=bool)
            for a, b in occ.get((radar, t), []):
                occupied[max(a, s0) - 1 : min(b, s1)] = True
            if ((base + need > cap[rtype[radar]]) & ~occupied).any():
                # 原码此处还有一段"往后挪发射时刻"的重试，因时间窗下标不更新而永远无效，
                # 等价于直接失败（见文件头注释第 1 条）
                fail[t, k:] = True
                continue

            pk, tl = tbl[row, 0], tbl[row, 1]
            tasks.append(
                Task(t, k, launcher, radar, row, float(pk), float(tl), float(t_kill))
            )
            kill_prob[t, k] = pk
            launcher_used[launcher] += 1
            new_slots = np.nonzero(~occupied)[0] + (s0 - 1)
            if len(new_slots):
                colsum[radar, new_slots] += need
                occ.setdefault((radar, t), []).append((s0, s1))
            last_kill_t[t] = t_kill
            last_radar[t] = radar

    survive = 1.0 - np.prod(1.0 - kill_prob, axis=1)
    kp = np.column_stack([kill_prob, survive])
    return EvalResult(
        obj=float(survive.sum()),
        kdg=np.array(kdg, dtype=np.int32).reshape(-1, 2),
        tasks=tasks,
        kill_prob=kp,
        fail=fail,
    )
