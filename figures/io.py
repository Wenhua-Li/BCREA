# -*- coding: utf-8 -*-
"""公共只读数据层（STYLE.md 第八节）：v3 批次的六个方法、30 实例、10 种子。

只读，不产图、不写批次。图脚本统一 `import style` + `io = style.load_io()`
（本文件与标准库同名，不能普通 import，见 style.load_io 的说明）；路径拼接、
方法名与文件名前缀的换算、实例短码解析都只写在这里，不在各图里复制。

批次布局（style.require_batch() 返回的目录下）：
    pools/<method_prefix>_<instance>_s<seed>.npz     主池（decs/objs/trace/kill_prob/config）
    scans/<method_prefix>_<instance>_s<seed>.json    16 格全网格扫描（cells.d30_g2 等）
    reopt/reopt_c<a>c<b>_<instance>_s<seed>.npz      真实模式重解（5 种子）
分析层在 data/results/<BATCH_V3>_analysis/*.csv，用 load_analysis(name) 读。

注意两个标签体系：
  - 池 npz 的 config["method"] 用长标签（nominal_kcea / pessimistic_resolve /
    sampled_worst_case / …），也是 style.METHOD 的键；
  - 落盘文件名用短前缀（nominal / pessimistic / nsga2 / sampled_wc / …）。
  load_pool/load_scan 会自动在两者之间换算并核对，图脚本只传长标签。

导入本模块会把项目根加进 sys.path，因此可以直接 `io.equiv_set(...)` 复用
core.robust_metrics 的等价集、贪心覆盖选择与鲁棒覆盖（图脚本不要再手写这三件套）。
"""
import json
import re
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd

import style

# ---------- 项目根进 sys.path（复用 core.robust_metrics，避免图脚本各写一份） ----------
if str(style.ROOT) not in sys.path:
    sys.path.insert(0, str(style.ROOT))

from core.robust_metrics import equiv_set, greedy_rc_select, robust_coverage  # noqa: E402

# ---------- 目录 ----------
POOLS_DIR = style.RESULTS / style.BATCH_V3 / "pools"
SCANS_DIR = style.RESULTS / style.BATCH_V3 / "scans"
REOPT_DIR = style.RESULTS / style.BATCH_V3 / "reopt"
ANALYSIS_DIR = style.RESULTS / f"{style.BATCH_V3}_analysis"

# ---------- 方法与规模 ----------
METHODS = tuple(style.METHOD_ORDER)          # 六个长标签，顺序即图例顺序
METHOD_PREFIX = {
    "nominal_kcea": "nominal",
    "pessimistic_resolve": "pessimistic",
    "nsga2": "nsga2",
    "sampled_worst_case": "sampled_wc",
    "bcrea_band": "bcrea_band",
    "bcrea_strict": "bcrea_strict",
}
PREFIX_METHOD = {v: k for k, v in METHOD_PREFIX.items()}
SCALES = (24, 48, 100)
SEEDS = tuple(range(1, 11))                  # pools/scans 各 10 种子
REOPT_SEEDS = tuple(range(1, 6))             # reopt 砍到 5 种子（1980 条）
BUDGET_PER_TARGET = 9000                     # 统一真实解码预算 D = 9000·N

Instance = namedtuple("Instance", "code n_target family spread")


def _build_instances() -> dict[str, Instance]:
    """30 个实例短码 → (N, family, spread)。口径同 experiments/make_tasks.py:36-42。"""
    out: dict[str, Instance] = {}
    for k in range(1, 11):
        out[f"i{k}"] = Instance(f"i{k}", 24, "tevc", "na")
    out["i11"] = Instance("i11", 48, "tevc", "na")
    out["i100"] = Instance("i100", 100, "tevc", "na")
    for n in SCALES:
        for mode in ("conc", "disp"):
            for rep in range(1, 4):
                code = f"gen{n}_{mode}_{rep:02d}"
                out[code] = Instance(code, n, "gen", mode)
    return out


INSTANCES = _build_instances()

_STEM_RE = re.compile(r"^(?P<inst>.+)_s(?P<seed>\d+)$")


def instances(n_target: int | None = None) -> tuple[str, ...]:
    """实例短码元组（默认按规模、家族、散布、编号排序，保证重跑顺序固定）。"""
    codes = [c for c, m in INSTANCES.items() if n_target is None or m.n_target == n_target]
    return tuple(sorted(codes, key=lambda c: (INSTANCES[c].n_target,
                                              INSTANCES[c].family,
                                              INSTANCES[c].spread, c)))


def instance_meta(code: str) -> Instance:
    if code not in INSTANCES:
        raise KeyError(f"[io] 未知实例短码: {code}")
    return INSTANCES[code]


def parse_stem(stem: str) -> tuple[str, str, int]:
    """文件名主干 → (方法长标签, 实例短码, 种子)。前缀按长度降序匹配，避免
    bcrea_band / bcrea_strict 之类的共享前缀误判。"""
    for prefix in sorted(PREFIX_METHOD, key=len, reverse=True):
        if stem.startswith(prefix + "_"):
            m = _STEM_RE.match(stem[len(prefix) + 1:])
            if not m:
                break
            return PREFIX_METHOD[prefix], m.group("inst"), int(m.group("seed"))
    raise ValueError(f"[io] 无法解析文件名主干: {stem}")


# ---------- 路径 ----------
def pool_path(method: str, instance: str, seed: int) -> Path:
    return POOLS_DIR / f"{METHOD_PREFIX[method]}_{instance}_s{seed:02d}.npz"


def scan_path(method: str, instance: str, seed: int) -> Path:
    return SCANS_DIR / f"{METHOD_PREFIX[method]}_{instance}_s{seed:02d}.json"


def reopt_path(instance: str, comps: tuple[int, int], seed: int) -> Path:
    a, b = comps
    return REOPT_DIR / f"reopt_c{a}c{b}_{instance}_s{seed:02d}.npz"


# ---------- 读取 ----------
def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"[io] 批次文件缺失: {path}")
    return path


def load_pool(method: str, instance: str, seed: int, check: bool = True) -> dict:
    """读一个主池 npz，返回 dict：decs/objs/trace/kill_prob/config(+派生字段)。

    check=True 时核对 config 的 method、n_target 与预算字段（STYLE.md 第七节）。"""
    z = np.load(_require(pool_path(method, instance, seed)), allow_pickle=False)
    cfg = json.loads(str(z["config"]))
    if check:
        if cfg.get("method") != method:
            raise ValueError(f"[io] 方法标签不符: 请求 {method}，文件 {cfg.get('method')}")
        meta = instance_meta(instance)
        if cfg.get("n_target") != meta.n_target:
            raise ValueError(f"[io] 目标数不符: {instance} 应为 {meta.n_target}，文件 {cfg.get('n_target')}")
        want = BUDGET_PER_TARGET * meta.n_target
        if cfg.get("decode_budget") != want:
            raise ValueError(f"[io] 预算不符: {cfg.get('decode_budget')} != {want}")
    return {
        "decs": np.asarray(z["decs"]),
        "objs": np.asarray(z["objs"], float),
        "trace": np.asarray(z["trace"], float),
        "kill_prob": np.asarray(z["kill_prob"], float),
        "config": cfg,
        "method": method,
        "instance": instance,
        "seed": int(seed),
        "n_target": instance_meta(instance).n_target,
        "pool_size": int(np.asarray(z["objs"]).size),
    }


def load_scan(method: str, instance: str, seed: int, check: bool = True) -> dict:
    """读一份扫描 JSON（16 格全网格）。扫描 JSON 无 method 字段，用 pool_provenance 核对。"""
    js = json.loads(_require(scan_path(method, instance, seed)).read_text(encoding="utf-8"))
    if check:
        prov = (js.get("pool_provenance") or {}).get("config") or {}
        if prov.get("method") != method:
            raise ValueError(f"[io] 扫描 provenance 方法标签不符: {prov.get('method')} != {method}")
        if prov.get("seed") != seed:
            raise ValueError(f"[io] 扫描 provenance 种子不符: {prov.get('seed')} != {seed}")
        meta = instance_meta(instance)
        if (js.get("config") or {}).get("n_target") != meta.n_target:
            raise ValueError(f"[io] 扫描目标数不符: {instance}")
    return js


def load_reopt(instance: str, comps: tuple[int, int] | list[int], seed: int, check: bool = True) -> dict:
    """读一份真实模式重解 npz（角点问题上的 KCEA 重解）。"""
    z = np.load(_require(reopt_path(instance, tuple(comps), seed)), allow_pickle=False)
    cfg = json.loads(str(z["config"]))
    if check:
        meta = instance_meta(instance)
        if cfg.get("n_target") != meta.n_target:
            raise ValueError(f"[io] reopt 目标数不符: {instance}")
        if list(cfg.get("components") or []) != list(comps):
            raise ValueError(f"[io] reopt 分量不符: {cfg.get('components')} != {list(comps)}")
        if cfg.get("seed") != seed:
            raise ValueError(f"[io] reopt 种子不符: {cfg.get('seed')} != {seed}")
    return {
        "decs": np.asarray(z["decs"]),
        "objs": np.asarray(z["objs"], float),
        "trace": np.asarray(z["trace"], float),
        "config": cfg,
        "instance": instance,
        "comps": tuple(comps),
        "seed": int(seed),
    }


def load_analysis(name: str) -> pd.DataFrame:
    """读分析层 CSV（name 可带或不带 .csv）。"""
    fname = name if name.endswith(".csv") else f"{name}.csv"
    return pd.read_csv(_require(ANALYSIS_DIR / fname))


def scan_arrays(scan: dict, cell: str = "d30_g2") -> dict:
    """把一格的 JSON 列表转成 numpy：objsS (n_scen, n_pool)、deg、objs0、eq_idx、masks。"""
    c = scan["cells"][cell]
    nom = scan["nominal"]
    return {
        "cell": c,
        "key": cell,
        "objsS": np.asarray(c["objsS"], float),
        "deg": np.asarray(c["deg"], float),
        "objs_wc": np.asarray(c["objs_wc"], float),
        "masks": np.asarray(c["masks"], int),
        "objs0": np.asarray(nom["objs0"], float),
        "eq_idx": np.asarray(nom["eq_idx"], int),
        "best": float(nom["best"]),
        "band_rel": float((scan.get("config") or {}).get("band_rel", np.nan)),
    }
