"""防御方参数与 SWTA 问题构建（前作 ParameterSetting.m + CalMissileSeeTarget.m 的移植）。

一个 SWTA 问题 = 目标数据（EnemyInfo，来自 .mat）+ 防御方部署参数（DefenseParams）。
构建时对每个目标预计算：
  - 可选组合 combination：哪些（发射车, 雷达）对能打到它，附最大命中概率；
  - 命中概率表 pkill：每对（发射车, 雷达）每个轨迹时段的
    [命中概率, 发射时刻, 当前飞行时刻, 导弹飞行时长]，按命中概率降序、发射时刻升序排好，
    尾部全零行已截断（解码器据此知道走到头了）；
  - abandon：完全打不到的目标。

时间窗、弹药限量等约束参数也在这里（解码器只读）。

口径说明（与代码-论文差异有关，2026-10-02 移植时核定）：
  - 前作 MATLAB 默认部署：6 雷达 / 6 发射车 / 每车 15 弹 / 每雷达 10 通道；
    论文 MADRA-LIB 各实例的部署不同（P3 是 6/6/5/5），要复现某个实例就用
    DefenseParams(...) 覆盖对应字段。
  - MATLAB 的雷达方位角判断没有做 ±180° 环绕（abs(角差)<Theta 直接比），
    本文件照抄此行为，改了会和前作数字对不上。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from core.scenarios import EnemyInfo

# 前作 ParameterSetting.m 里的常量（rKill/pKill 六种雷达×六种导弹全部同值，只存一份）
# rKill 行=目标类型 1..3，列=5 个拦截远/近界半径（公里）
R_KILL = np.array(
    [
        [1.0, 3.0, 11.0, 17.0, 21.0],
        [1.0, 3.0, 17.0, 22.0, 25.0],
        [1.0, 3.0, 11.0, 12.0, 18.0],
    ]
)
# pKill 行=目标类型 1..3，列=轨迹落入第几个拦截环（1..4）时的基础命中概率
P_KILL = np.array(
    [
        [0.0, 0.3, 0.5, 0.4, 0.2, 0.0],
        [0.0, 0.4, 0.6, 0.3, 0.3, 0.0],
        [0.0, 0.5, 0.7, 0.2, 0.1, 0.0],
    ]
)

_DEFAULT_RADAR_LOCATION = np.array(
    [[-5.0, 0.0], [5.0, 0.0], [-5.0, 10.0], [5.0, 10.0], [-5.0, 20.0], [5.0, 20.0]]
)
_DEFAULT_RADAR_ANGLE = np.array([3 * math.pi / 5, 2 * math.pi / 5, math.pi / 2, math.pi / 2, 2 * math.pi / 3, math.pi / 3])


@dataclass(frozen=True)
class DefenseParams:
    """防御方部署与约束参数。默认值 = 前作 MATLAB ParameterSetting.m 的写死值。"""

    dT: float = 0.8  # 时间离散步长（秒）
    preT: float = 15.0  # 命中前需要的连续跟踪时长（秒）
    max_missiles: int = 3  # 每个目标最多几发（Target.MaxMissiles）
    allow_multi: bool = True  # false 时每目标只允许一发
    mode_fire: int = 2  # 1=集火（相邻命中间隔 [0,5]），2=分火（[10,15]）
    delta_t2: float = 10.0
    delta_t3: float = 15.0

    # 雷达（默认 6 部，1 部 64km 探程的 1 型 + 5 部 2 型）
    radar_location: np.ndarray = field(default_factory=lambda: _DEFAULT_RADAR_LOCATION.copy())
    radar_angle: np.ndarray = field(default_factory=lambda: _DEFAULT_RADAR_ANGLE.copy())  # 法线朝向（弧度）
    radar_type: np.ndarray = field(default_factory=lambda: np.array([1, 2, 2, 2, 2, 2]))
    radar_R: np.ndarray = field(default_factory=lambda: np.array(  # 雷达类型×目标类型的探测距离
        [[64.0, 90.0, 36.0], [32.0, 45.0, 18.0], [32.0, 45.0, 18.0], [32.0, 45.0, 18.0], [32.0, 45.0, 18.0], [32.0, 45.0, 18.0]]
    ))
    radar_accuracy: np.ndarray = field(default_factory=lambda: np.array([3.0] * 6))  # 按雷达类型
    radar_n_channel: np.ndarray = field(default_factory=lambda: np.array([10.0] * 6))  # 按雷达类型
    radar_theta: np.ndarray = field(default_factory=lambda: np.array([70.0] * 6))  # 探测半张角（度），按雷达类型
    radar_coefficient: np.ndarray = field(default_factory=lambda: np.ones((6, 3)))  # 雷达类型×目标类型的通道占用系数

    # —— 扰动钩子（2026-10-02 增量新增，默认 None = 行为与之前完全一致，见总账第十七笔）——
    # 按"部"的探测威力 (nR,3)，设置后优先于 radar_R 的按类型查表；用于逐部缩水的不确定扰动
    radar_R_unit: np.ndarray | None = None
    # 按"发射车"的杀伤边界缩放 (nW,)，乘在 R_KILL 半径上；用于逐车缩水的不确定扰动
    rkill_scale_unit: np.ndarray | None = None

    # 发射车（默认 6 部，各带 1..6 型导弹）
    missile_location: np.ndarray = field(default_factory=lambda: _DEFAULT_RADAR_LOCATION.copy())  # 前作里发射车与雷达同位部署
    missile_type: np.ndarray = field(default_factory=lambda: np.array([1, 2, 3, 4, 5, 6]))
    missile_v: np.ndarray = field(default_factory=lambda: np.array([0.5] * 6))  # 按导弹型号
    missile_theta: float = 70.0  # 发射方向与目标飞行方向的夹角上限（度）
    missile_load_limit: np.ndarray = field(default_factory=lambda: np.array([15] * 6))  # 按导弹型号的每车弹数上限

    # 目标类型参数
    target_v: np.ndarray = field(default_factory=lambda: np.array([0.2, 0.1, 0.05]))  # 三类目标的速度
    target_cost: np.ndarray = field(default_factory=lambda: np.array([20.0, 10.0, 1.0]))  # 三类目标的价值

    def __post_init__(self) -> None:
        if not self.allow_multi:
            object.__setattr__(self, "max_missiles", 1)

    @property
    def delta_t1(self) -> tuple[float, float]:
        lo, hi = (0.0, 5.0) if self.mode_fire == 1 else (10.0, 15.0)
        return lo, hi

    @property
    def n_radar(self) -> int:
        return len(self.radar_location)

    @property
    def n_missile(self) -> int:
        return len(self.missile_location)

    @property
    def pre_nt(self) -> int:
        return math.ceil(self.preT / self.dT)


class TargetModel:
    """单个目标的预处理结果（CalMissileSeeTarget 对一个目标的输出）。

    pkill[(发射车下标, 雷达下标)] 是 (m,4) 数组：列 = [P, 发射时刻, 飞行时刻, 导弹飞行时长]，
    只保留 P>0 的行（原表尾部全零行截断，走到头即不可行）。
    """

    __slots__ = ("T", "nP", "combination", "pkill")

    def __init__(self, T: int, nP: int, combination: np.ndarray, pkill: dict[tuple[int, int], np.ndarray]):
        self.T = T  # 目标总飞行时长（秒）
        self.nP = nP  # 轨迹离散步数
        self.combination = combination  # (k,3)：[发射车下标, 雷达下标, 最大命中概率]
        self.pkill = pkill


class SWTAProblem:
    """一个完整的多解 SWTA 问题实例（解码器与算法只读这个）。"""

    def __init__(self, enemy: EnemyInfo, params: DefenseParams | None = None):
        self.enemy = enemy
        self.params = params or DefenseParams()
        p = self.params

        n = enemy.n_targets
        self.n_target = n
        self.max_missiles = p.max_missiles
        self.nS = 0  # 全局时间槽数，建完目标后取最大
        self.ttype = enemy.type_  # (N,) 1..3

        # 每目标的飞行时长（秒）与轨迹步数（前作口径：T=ceil(length/v)，nP=floor(T/dT)）
        length = enemy.start[:, 1] / np.sin(enemy.angle)
        speed = p.target_v[enemy.type_ - 1]
        self.flight_T = np.ceil(length / speed).astype(int)
        self.nP = (self.flight_T / p.dT).astype(int)

        self.targets: list[TargetModel] = []
        for i in range(n):
            self.targets.append(self._build_target(i))

        self.nS = int(self.nP.max())
        # 解码顺序：飞行时间短的目标先处理（前作 targetInd）
        self.target_ind = np.argsort(self.flight_T, kind="stable")
        self.abandon = np.array([t.combination.shape[0] == 0 for t in self.targets])

    # ---- 单目标预处理（CalMissileSeeTarget 的向量化移植，静态口径 curT=1） ----

    def _build_target(self, i: int) -> TargetModel:
        p = self.params
        nP = int(self.nP[i])
        t_type = int(self.ttype[i]) - 1  # 0 基目标类型
        ang = float(self.enemy.angle[i])

        # 轨迹：目标朝 x 轴飞，方向 = -[cos, sin](angle)
        t_axis = np.arange(1, nP + 1) * p.dT  # 各步的飞行时刻（秒）
        cur_l = t_axis * p.target_v[t_type]
        track = self.enemy.start[i][None, :] + cur_l[:, None] * np.array([-math.cos(ang), -math.sin(ang)])[None, :]

        # 雷达可见性（前作没做 ±180° 环绕，照抄）
        dist_r = np.linalg.norm(track[:, None, :] - p.radar_location[None, :, :], axis=2)  # (nP, nR)
        vec_r = track[:, None, :] - p.radar_location[None, :, :]
        angle_deg = np.degrees(np.arctan2(vec_r[:, :, 1], vec_r[:, :, 0]))
        radar_type_idx = p.radar_type - 1
        if p.radar_R_unit is not None:  # 扰动钩子：按部探测威力优先
            R_eff = p.radar_R_unit[:, t_type]
        else:
            R_eff = p.radar_R[radar_type_idx][:, t_type]
        see_r = (
            (np.abs(angle_deg - np.degrees(p.radar_angle)[None, :]) < p.radar_theta[radar_type_idx][None, :])
            & (dist_r < R_eff[None, :])
        )  # (nP, nR)

        # 发射车可达性：数落入几个拦截环（5=进最小内界，视为不可打）
        dist_m = np.linalg.norm(track[:, None, :] - p.missile_location[None, :, :], axis=2)  # (nP, nW)
        ring = np.zeros((nP, p.n_missile))
        rk_scale = p.rkill_scale_unit[None, :] if p.rkill_scale_unit is not None else 1.0  # 扰动钩子
        for c in range(5):
            ring += (dist_m < R_KILL[t_type, c] * rk_scale).astype(float)
        # 发射方向与目标飞行方向夹角超过 missile_theta 的位置不可打（norm=0 时 MATLAB 得 NaN，
        # NaN>70 为假，等效于不剔除；照抄）
        vec_m = track[:, None, :] - p.missile_location[None, :, :]
        dirv = np.array([math.cos(ang), math.sin(ang)])
        norms = np.linalg.norm(vec_m, axis=2)
        cosang = np.where(norms > 0, (vec_m @ dirv) / np.where(norms > 0, norms, 1.0), np.nan)
        ang_m = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
        ring[ang_m > p.missile_theta] = 0.0
        ring[ring == 5.0] = 0.0  # (nP, nW)，0..4 环

        # 命中概率表
        pkill: dict[tuple[int, int], np.ndarray] = {}
        flag_atk_s = np.zeros((p.n_missile, p.n_radar))
        radar_acc = p.radar_accuracy[radar_type_idx]  # (nR,)
        for j in range(p.n_missile):
            d = dist_m[:, j]
            fly_t = d / p.missile_v[p.missile_type[j] - 1]
            launch_t = t_axis - fly_t
            for k in range(p.n_radar):
                tmp_p = ring[:, j] * see_r[:, k]
                # 命中精度修正：雷达 k 的标定精度按"目标-发射车距离"分段（前作口径）
                ra = radar_acc[k]
                ca = np.full(nP, ra)
                ca = np.where(d <= 10, ra - 2.0, ca)
                ca = np.where((d > 10) & (d <= 20), ra - 1.0, ca)
                ca = np.where(d > 30, ca + np.abs(d - 100.0) / 30.0, ca)
                P = np.minimum(tmp_p * ra / ca, 0.8)
                P[tmp_p == 0] = 0.0
                # 命中前 preT 内必须可跟踪（前作用同一 (j,k) 的 P 全非零做近似），且发射时刻不能为负
                infeasible = np.zeros(nP, dtype=bool)
                if nP > p.pre_nt:
                    zero_cum = np.concatenate([[0], np.cumsum((P == 0).astype(int))])
                    idx = np.arange(p.pre_nt + 1, nP + 1)
                    has_zero = zero_cum[idx] - zero_cum[idx - (p.pre_nt + 1)] > 0
                    infeasible[p.pre_nt:] = has_zero
                infeasible[: p.pre_nt] = True
                infeasible |= launch_t < 0
                if infeasible.any():
                    P[infeasible] = 0.0
                flag_atk_s[j, k] = P.max() if nP else 0.0
                if P.max() <= 0:
                    continue
                table = np.column_stack([P, np.where(infeasible, 0.0, launch_t), t_axis, fly_t])
                order = np.lexsort((table[:, 1], -table[:, 0]))  # P 降序，发射时刻升序（稳定排序）
                table = table[order]
                pos_rows = int((table[:, 0] > 0).sum())
                pkill[(j, k)] = table[:pos_rows]

        ms, rs = np.nonzero(flag_atk_s > 0)
        combination = np.column_stack([ms, rs, flag_atk_s[ms, rs]])
        return TargetModel(T=int(self.flight_T[i]), nP=nP, combination=combination, pkill=pkill)
