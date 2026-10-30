"""自建中大规模场景生成器（评审 P3/B#2：拆"48/100 目标各单实例"的统计地基炸弹）。

口径贴近前作 TEVC 实例几何（实测 Info_enemy_1/11/100）：
  x ∈ [-32.4, 32]、y ∈ [12.4, 40]、angle ∈ [0.65, 2.51] rad、三类型近似均匀。
密度两档（把"容量 slack"做成可控变量而不是巧合）：
  tight：x 收窄到 [-12, 12]（目标扎堆 → 通道/弹药竞争紧）
  loose：x 全带 [-32.4, 32]（分散 → slack）
可行性校验（core 构建后）：abandon 全 False、每目标可选组合数 ≥ 2、并记录名义饱和标志
（KCEA 快速跑 dev 口径看最优是否顶到"全目标摧毁"上界附近，仅作元数据记录）。

产物：
  默认（--suite v1）：data/scenarios/generated_v1/g{N}{t|l}.mat（前作同格式 Info_enemy 结构）
    + manifest.json（种子/参数/校验统计——AGENTS.md 第 6 条：生成脚本+种子存档保证可复现）。
    行为与 2026-10-03 首版完全一致，不覆盖历史实例。
  --suite v3：data/scenarios/generated/，24/48/100 × concentrated/dispersed × 3 独立实例共 18 个
    （gen24_conc_01 式代码命名，不进论文），种子 8001 起顺序分配，manifest 记重试轮数等。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import savemat

from core.model import DefenseParams, SWTAProblem
from core.scenarios import EnemyInfo

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "scenarios" / "generated_v1"
OUT_DIR_V3 = Path(__file__).resolve().parents[1] / "data" / "scenarios" / "generated"

# (实例名, N, 密度档, 种子)
INSTANCES = [
    ("g48t", 48, "tight", 7001),
    ("g48l", 48, "loose", 7002),
    ("g100t", 100, "tight", 7003),
    ("g100l", 100, "loose", 7004),
]
X_BAND = {"tight": (-12.0, 12.0), "loose": (-32.4, 32.0)}

# v3 套件：三规模 × 两种散布 × 3 个独立实例；密度档对应 tight/loose 的 x 带
V3_N = (24, 48, 100)
V3_MODES = (("conc", "tight"), ("disp", "loose"))   # (v3 名字段, X_BAND 档)
V3_REPS = 3
V3_SEED0 = 8001


def _draw_repair(n: int, mode: str, seed: int, max_rounds: int = 5):
    """逐目标拒绝采样：先整批采 n 个目标，打不到的目标（组合数<2）只重采该行，最多 max_rounds 轮。
    返回 (enemy, 实际重试轮数, 修复后仍不合格的目标下标)。抽随机数的顺序与 2026-10-03 首版一致。"""
    rng = np.random.default_rng(seed)
    x_lo, x_hi = X_BAND[mode]

    def draw(m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.column_stack([rng.uniform(x_lo, x_hi, m), rng.uniform(12.4, 40.0, m)]),
            rng.uniform(0.65, 2.51, m),
            rng.integers(1, 4, m),
        )

    start, angle, type_ = draw(n)
    rounds = 0
    for r in range(max_rounds):
        enemy = EnemyInfo(start=start, angle=angle, type_=type_, source=Path(f"gen:{mode}:{seed}"))
        prob = SWTAProblem(enemy, DefenseParams())
        bad = np.nonzero(np.array([t.combination.shape[0] for t in prob.targets]) < 2)[0]
        if len(bad) == 0:
            return enemy, rounds, bad
        s2, a2, t2 = draw(len(bad))
        start[bad], angle[bad], type_[bad] = s2, a2, t2
        rounds = r + 1
    enemy = EnemyInfo(start=start, angle=angle, type_=type_, source=Path(f"gen:{mode}:{seed}"))
    prob = SWTAProblem(enemy, DefenseParams())
    bad = np.nonzero(np.array([t.combination.shape[0] for t in prob.targets]) < 2)[0]
    return enemy, rounds, bad


def sample_instance(n: int, mode: str, seed: int) -> EnemyInfo:
    """v1 套件的旧接口（行为不变）：拒绝采样修复后返回，仍不合格由 validate 把关。"""
    enemy, _, _ = _draw_repair(n, mode, seed)
    return enemy


def validate(enemy: EnemyInfo) -> dict:
    prob = SWTAProblem(enemy, DefenseParams())
    n_comb = np.array([t.combination.shape[0] for t in prob.targets])
    return {
        "n_target": int(prob.n_target),
        "n_abandon": int(prob.abandon.sum()),
        "comb_min": int(n_comb.min()),
        "comb_mean": float(n_comb.mean()),
        "nS": int(prob.nS),
    }


def _save(enemy: EnemyInfo, mat: Path) -> None:
    mat.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        mat,
        {"Info_enemy": {
            "enemy_start": enemy.start,
            "enemy_angle": enemy.angle,
            "enemy_type": enemy.type_,
        }},
    )


def run_v1(out_dir: Path) -> None:
    """默认套件：历史上那 4 个实例，行为与首版一致（不覆盖就校验，manifest 原样）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, n, mode, seed in INSTANCES:
        enemy = sample_instance(n, mode, seed)
        stats = validate(enemy)
        if stats["n_abandon"] > 0 or stats["comb_min"] < 2:
            raise SystemExit(f"{name} 校验不过：{stats}（换种子重生成）")
        mat = out_dir / f"{name}.mat"
        _save(enemy, mat)
        manifest[name] = {"n": n, "mode": mode, "seed": seed, **stats}
        print(f"{name}: N={n} {mode} seed={seed} abandon={stats['n_abandon']} "
              f"comb∈[{stats['comb_min']},{stats['comb_mean']:.1f}] -> {mat.name}")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("manifest 已存档")


def run_v3(out_dir: Path) -> None:
    """v3 套件：18 个独立实例（种子 8001 起顺序分配），失败即报错，不按效果挑种子。
    拒绝采样上限 50 轮（v1 是 5 轮；只是同一规则的确定性重试，不换种子、不挑效果）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    seed = V3_SEED0
    for n in V3_N:
        for mode_tag, band in V3_MODES:
            for rep in range(1, V3_REPS + 1):
                name = f"gen{n}_{mode_tag}_{rep:02d}"
                enemy, rounds, bad = _draw_repair(n, band, seed, max_rounds=50)
                if len(bad) > 0:
                    raise SystemExit(f"{name} 重试 {rounds} 轮后仍有打不到的目标 {bad.tolist()}，"
                                     f"按规矩报错（不换种子挑效果）")
                stats = validate(enemy)
                if stats["n_abandon"] > 0 or stats["comb_min"] < 2:
                    raise SystemExit(f"{name} 校验不过：{stats}")
                mat = out_dir / f"{name}.mat"
                _save(enemy, mat)
                manifest[name] = {
                    "n": n, "mode": mode_tag, "seed": seed,
                    "x_range": [round(float(enemy.start[:, 0].min()), 3),
                                round(float(enemy.start[:, 0].max()), 3)],
                    "y_range": [round(float(enemy.start[:, 1].min()), 3),
                                round(float(enemy.start[:, 1].max()), 3)],
                    "type_counts": {str(t): int((enemy.type_ == t).sum()) for t in (1, 2, 3)},
                    "retry_rounds": int(rounds),
                    **stats,
                }
                print(f"{name}: N={n} {mode_tag} seed={seed} 重试={rounds} "
                      f"abandon={stats['n_abandon']} comb∈[{stats['comb_min']},{stats['comb_mean']:.1f}] "
                      f"-> {mat.name}", flush=True)
                seed += 1
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest 已存档（{len(manifest)} 个实例）")


def main() -> None:
    ap = argparse.ArgumentParser(description="场景生成器（默认 v1 四实例；--suite v3 生成 18 实例新套件）")
    ap.add_argument("--suite", default="v1", choices=["v1", "v3"])
    ap.add_argument("--out-dir", default=None, help="输出目录（默认 v1→scenarios_gen，v3→scenarios_gen_v3）")
    args = ap.parse_args()
    if args.suite == "v3":
        run_v3(Path(args.out_dir) if args.out_dir else OUT_DIR_V3)
    else:
        run_v1(Path(args.out_dir) if args.out_dir else OUT_DIR)


if __name__ == "__main__":
    main()
