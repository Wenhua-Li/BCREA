"""v3 统一实验任务清单生成器（2026-10-13，已批准方案"锁配置"阶段产物）。

只生成清单，不启动任务。三套清单 + manifest：

  pools  主池：30 实例 × 6 方法 × 10 种子 = 1800 条
  scans  扫描：每个池一条 exp_robust（δ∈{0.1,0.2,0.3,0.4} × Γ∈{1,2,3,12} 全网格，
         主单元 d30_g2 在其中）= 1800 条
  reopt  真实模式重解框架：6 代表实例 × C(12,2)=66 个 Γ=2 模式 × **5 种子 = 1980 条**
         （2026-10-15 拍板从 10 种子砍到 5 种子，执行清单见 tasks_20261013_v3_reopt_5seed.txt；
         复用 exp_corner_pool.py：在实现的扰动模式问题上用 KCEA 重求解）。
         注意 reopt_tasks() 仍按 10 种子生成基础清单（3960 条），测试核对的是这个基础口径；
         正式批次用的 5 种子子集，两者不可混为一批。

统一口径（可复现、预算可比、标签不可混）：
  - 共同真实解码预算 D = 9000·N（N=24/48/100 → 216000/432000/900000）；
    KCEA 系每个体 1 次解码 → --fes 9000x；BCREA/NSGA-II K=8（9×FE）、sampled_wc M=16（17×FE）
    → --decode-budget D（按真实解码量卡终止，不超支）。
  - BCREA/NSGA-II：K=8、scen_seed=9001、分层采样、δ=.3、Γ=2、eps=.005；
    sampled_wc：M=16、scen_seed=9101、分层。
  - 实例：原 12 个 TEVC（Info_enemy_1..10=24 目标、11=48、100=100，服务器相对路径
    data/scenarios/tevc/...）+ v3 新生成 18 个（data/scenarios/generated/gen*_*.mat，种子 8001 起）。

跑法（仓库根）：
  PYTHONPATH=. python experiments/make_tasks.py     # 生成 data/tasks/tasks_20261013_v3_*.txt + manifest
"""
from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path

from experiments.gen_scenarios import V3_MODES, V3_N, V3_REPS

TASKS_DIR = Path(__file__).resolve().parents[1] / "data" / "tasks"
PY = "$PYTHON"

# ---- 实例表：代码 ID -> (场景文件, N) ----
INSTANCES: dict[str, tuple[str, int]] = {
    **{f"i{k}": (f"data/scenarios/tevc/Info_enemy_{k}.mat", 24) for k in range(1, 11)},
    "i11": ("data/scenarios/tevc/Info_enemy_11.mat", 48),
    "i100": ("data/scenarios/tevc/Info_enemy_100.mat", 100),
    **{f"gen{n}_{mode}_{rep:02d}": (f"data/scenarios/generated/gen{n}_{mode}_{rep:02d}.mat", n)
       for n in V3_N for mode, _ in V3_MODES for rep in range(1, V3_REPS + 1)},
}

# reopt 代表实例：三规模 × 两种散布各取 1 号（都在 v3 新套件里，覆盖密度变量）
REOPT_INSTANCES = ["gen24_conc_01", "gen24_disp_01",
                   "gen48_conc_01", "gen48_disp_01",
                   "gen100_conc_01", "gen100_disp_01"]

SEEDS = range(1, 11)
DELTA, GAMMA = 0.3, 2
RES = "data/results/20261013_v3"
N_COMP = 12          # 不确定分量数（6 雷达 + 6 发射车）
REOPT_GAMMA = 2      # 真实模式 = Γ=2 的 C(12,2)=66 个分量对


def budget_of(inst: str) -> int:
    """共同真实解码预算 D = 9000·N。"""
    return 9000 * INSTANCES[inst][1]


def pool_tasks() -> list[tuple[str, dict]]:
    """主池任务行。返回 (命令行, manifest 行) 列表。"""
    rows = []
    for inst, (mat, _n) in INSTANCES.items():
        D = budget_of(inst)
        for s in SEEDS:
            jobs = {
                "nominal": (
                    f"PYTHONPATH=. {PY} core/kcea.py --scenario {mat} "
                    f"--seed {s} --fes 9000x --selection matlab "
                    f"--out {RES}/pools/nominal_{inst}_s{s:02d}.npz"),
                "pessimistic": (
                    f"PYTHONPATH=. {PY} experiments/run_pessimistic_pool.py --scenario {mat} "
                    f"--seed {s} --delta {DELTA} --fes 9000x "
                    f"--out {RES}/pools/pessimistic_{inst}_s{s:02d}.npz"),
                "nsga2": (
                    f"PYTHONPATH=. {PY} experiments/run_nsga2_pool.py --scenario {mat} "
                    f"--seed {s} --decode-budget {D} --delta {DELTA} --gamma {GAMMA} "
                    f"--n-scen 8 --scen-seed 9001 "
                    f"--out {RES}/pools/nsga2_{inst}_s{s:02d}.npz"),
                "sampled_wc": (
                    f"PYTHONPATH=. {PY} experiments/run_sampled_wc.py --scenario {mat} "
                    f"--seed {s} --decode-budget {D} --m-scen 16 --delta {DELTA} "
                    f"--gamma {GAMMA} --scen-seed 9101 "
                    f"--out {RES}/pools/sampled_wc_{inst}_s{s:02d}.npz"),
                "bcrea_band": (
                    f"PYTHONPATH=. {PY} core/bcrea.py --scenario {mat} "
                    f"--seed {s} --decode-budget {D} --delta {DELTA} --gamma {GAMMA} "
                    f"--n-scen 8 --scen-seed 9001 --eps-band 0.005 --protocol v3_equal_decode "
                    f"--out {RES}/pools/bcrea_band_{inst}_s{s:02d}.npz"),
                "bcrea_strict": (
                    f"PYTHONPATH=. {PY} core/bcrea.py --scenario {mat} "
                    f"--seed {s} --decode-budget {D} --delta {DELTA} --gamma {GAMMA} "
                    f"--n-scen 8 --scen-seed 9001 --eps-band 0.005 --protocol v3_equal_decode "
                    f"--no-downgrade "
                    f"--out {RES}/pools/bcrea_strict_{inst}_s{s:02d}.npz"),
            }
            for method, cmd in jobs.items():
                rows.append((cmd, {"kind": "pool", "instance": inst, "method": method,
                                   "seed": s, "decode_budget": D,
                                   "out": f"{RES}/pools/{method}_{inst}_s{s:02d}.npz"}))
    return rows


def scan_tasks() -> list[tuple[str, dict]]:
    """扫描任务行：每个池一条 exp_robust 全网格扫描。"""
    rows = []
    for inst, (mat, _n) in INSTANCES.items():
        for method in ("nominal", "pessimistic", "nsga2", "sampled_wc", "bcrea_band", "bcrea_strict"):
            for s in SEEDS:
                pool = f"{RES}/pools/{method}_{inst}_s{s:02d}.npz"
                out = f"{RES}/scans/{method}_{inst}_s{s:02d}.json"
                cmd = (f"PYTHONPATH=. {PY} experiments/run_pool_scan.py --scenario {mat} "
                       f"--pool {pool} --deltas 0.1,0.2,0.3,0.4 --gammas 1,2,3,12 "
                       f"--n-scen 60 --enum-cap 250 --scen-seed 6000 --out {out}")
                rows.append((cmd, {"kind": "scan", "instance": inst, "method": method,
                                   "seed": s, "decode_budget": "", "out": out}))
    return rows


def reopt_tasks() -> list[tuple[str, dict]]:
    """真实模式重解任务行：66 个 Γ=2 分量对 × 6 代表实例 × 10 种子，KCEA 重求解。"""
    rows = []
    pairs = list(combinations(range(N_COMP), REOPT_GAMMA))
    for inst in REOPT_INSTANCES:
        mat, _n = INSTANCES[inst]
        D = budget_of(inst)
        for a, b in pairs:
            comps = f"{a},{b}"
            for s in SEEDS:
                out = f"{RES}/reopt/reopt_c{a}c{b}_{inst}_s{s:02d}.npz"
                cmd = (f"PYTHONPATH=. {PY} experiments/run_corner_pool.py --scenario {mat} "
                       f"--components {comps} --seed {s} --delta {DELTA} --fes 9000x "
                       f"--out {out}")
                rows.append((cmd, {"kind": "reopt", "instance": inst,
                                   "method": "true_reoptimization", "seed": s,
                                   "decode_budget": D, "out": out}))
    return rows


def main() -> None:
    groups = [("pools", pool_tasks), ("scans", scan_tasks), ("reopt", reopt_tasks)]
    manifest: list[dict] = []
    for tag, fn in groups:
        rows = fn()
        fp = TASKS_DIR / f"tasks_20261013_v3_{tag}.txt"
        fp.write_text("\n".join(cmd for cmd, _ in rows) + "\n", encoding="utf-8")
        manifest.extend(meta for _, meta in rows)
        print(f"{fp.name}: {len(rows)} 任务")
    (TASKS_DIR / "tasks_20261013_v3_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(TASKS_DIR / "tasks_20261013_v3_manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["kind", "instance", "method", "seed", "decode_budget", "out"])
        w.writeheader()
        w.writerows(manifest)
    print(f"manifest：{len(manifest)} 行（json+csv）")


if __name__ == "__main__":
    main()
