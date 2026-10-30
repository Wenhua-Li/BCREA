"""读前作 .mat 场景文件里的目标（敌方）信息。

前作的场景文件只存目标数据（起始位置、类型、航向角）；雷达、导弹等防御方
参数在前作 ParameterSetting.m 里写死，归 model.py 管，这里不碰。

用法：
    from core.scenarios import load_enemy, load_scenarios
    enemy = load_enemy(r"...\Scenario\Info_enemy_1.mat")
    all12 = load_scenarios(r"...\Scenario")

命令行验证（打印每个场景的目标数和字段检查结果）：
    python core/scenarios.py <场景文件夹>
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

# 前作场景文件的变量名和字段名（Init.m / ParameterSetting.m 的读取口径）
MAT_VAR = "Info_enemy"
FIELDS = {"enemy_start": "起始坐标 (N,2)", "enemy_angle": "航向角 (N,)", "enemy_type": "目标类型 (N,)"}


@dataclass(frozen=True)
class EnemyInfo:
    """一个场景的目标信息。start 单位按前作口径（公里），angle 为弧度。"""

    start: np.ndarray  # (N,2) float
    angle: np.ndarray  # (N,)  float
    type_: np.ndarray  # (N,)  整数，取值 1..nTargetType（前作默认 3 类）
    source: Path

    @property
    def n_targets(self) -> int:
        return int(self.start.shape[0])


def load_enemy(mat_path: str | Path) -> EnemyInfo:
    """读单个场景文件，字段缺失或形状不对时抛 ValueError（带文件名）。"""
    mat_path = Path(mat_path)
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if MAT_VAR not in data:
        raise ValueError(f"{mat_path.name}: 缺变量 {MAT_VAR}，实际有 {[k for k in data if not k.startswith('__')]}")
    enemy = data[MAT_VAR]

    missing = [f for f in FIELDS if not hasattr(enemy, f)]
    if missing:
        raise ValueError(f"{mat_path.name}: {MAT_VAR} 缺字段 {missing}，实际有 {enemy._fieldnames}")

    start = np.asarray(enemy.enemy_start, dtype=float)
    angle = np.asarray(enemy.enemy_angle, dtype=float)
    type_ = np.asarray(enemy.enemy_type)

    if start.ndim != 2 or start.shape[1] != 2:
        raise ValueError(f"{mat_path.name}: enemy_start 形状 {start.shape}，应为 (N,2)")
    n = start.shape[0]
    if angle.shape != (n,) or type_.shape != (n,):
        raise ValueError(f"{mat_path.name}: 字段长度对不上（N={n}, angle{angle.shape}, type{type_.shape}）")
    if not np.isfinite(start).all() or not np.isfinite(angle).all():
        raise ValueError(f"{mat_path.name}: 位置或航向角里有非数值（NaN/Inf）")
    if (type_ < 1).any():
        raise ValueError(f"{mat_path.name}: 目标类型应从 1 开始，实际最小 {type_.min()}")

    return EnemyInfo(start=start, angle=angle, type_=type_.astype(int), source=mat_path)


def _natural_key(path: Path) -> list:
    """按文件名里的数字排序：Info_enemy_2 排在 Info_enemy_10 前面。"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def load_scenarios(folder: str | Path) -> dict[str, EnemyInfo]:
    """读一个文件夹里全部 Info_enemy_*.mat，返回 {文件名: EnemyInfo}，按编号自然排序。"""
    folder = Path(folder)
    files = sorted(folder.glob("Info_enemy_*.mat"), key=_natural_key)
    if not files:
        raise FileNotFoundError(f"{folder} 下没有 Info_enemy_*.mat")
    return {f.name: load_enemy(f) for f in files}


def verify(folder: str | Path, verbose: bool = True) -> bool:
    """逐个检查场景文件，打印汇总表。全部通过返回 True。"""
    scenarios = load_scenarios(folder)
    ok = True
    rows = []
    for name, e in scenarios.items():
        good = e.n_targets > 0 and e.start.shape == (e.n_targets, 2)
        ok = ok and good
        rows.append(
            f"{name:<20} N={e.n_targets:>4}  "
            f"type∈[{e.type_.min()},{e.type_.max()}]  "
            f"angle∈[{e.angle.min():+.2f},{e.angle.max():+.2f}]rad  "
            f"{'PASS' if good else 'FAIL'}"
        )
    if verbose:
        print(f"场景文件夹：{Path(folder).resolve()}")
        print(f"共 {len(rows)} 个场景，字段 {list(FIELDS)}")
        print("\n".join(rows))
        print("结论：", "全部通过" if ok else "存在问题，见 FAIL 行")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="验证前作 .mat 场景文件能否被 core 正确读取")
    ap.add_argument("folder", help="场景文件夹路径，例如前作的 Scenario/ 目录")
    args = ap.parse_args()
    sys.exit(0 if verify(args.folder) else 1)
