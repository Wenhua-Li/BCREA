# -*- coding: utf-8 -*-
"""BCREA 论文配图统一样式 v3（2026-10-13 整改方案 §8）。

用法（每个图脚本开头）：
    import style
    style.apply()          # 必须在 import matplotlib.pyplot 之前
    import matplotlib.pyplot as plt

规范细节见同目录 STYLE.md；这里是可执行的设计令牌：
- 字号：正文 9pt，地板 8pt（刻度/图例），面板标签 10pt 加粗——低于地板重排版，不缩字；
- 方法色全篇固定（METHOD）：band/strict 同蓝色族、实线/虚线区分；nominal 灰；基线固定色；
- 线宽层级：数据 1.2 / 强调 1.8 / 参考 0.8 虚线；
- 坐标：去上/右边框，仅 y 向细网格；图例无框；
- 尺寸：单栏 88.9mm / 跨栏 182mm（IEEE 双栏）；
- 输出：PDF 进 paper/figures/，PNG 预览进 code/figure/build/。

数据铁律：性能图只读 data/results/ 本轮锁定批次，脚本开头用 require_batch() 核对；
示意图（fig1/fig3）不读实验数据。正确解释器是 py -3.11。
"""
from pathlib import Path
import sys

import matplotlib as mpl

# ---------- 路径（全部相对项目根，禁止硬编码开发机路径） ----------
ROOT = Path(__file__).resolve().parents[1]           # 项目根
RESULTS = ROOT / "data" / "results"                  # 实验批次（只读）
OUT_DIR = ROOT / "paper" / "figures"                 # 最终 PDF（进 tex）
QA_DIR = ROOT / "figures" / "build"                  # PNG 预览与 QA 产物

# 本轮锁定的正式批次（2026-10-13 v3 协议；批次生成后更新此处并全图重跑）
BATCH_V3 = "20261013_v3"

# ---------- 尺寸（mm→inch） ----------
MM = 1.0 / 25.4
SINGLE_COL = 88.9 * MM    # 单栏图宽（IEEE 3.5in）
DOUBLE_COL = 182 * MM     # 跨栏图宽（IEEE 7.16in，figure*）

# ---------- Okabe-Ito 色盲安全色板 ----------
OI = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermillion": "#D55E00", "purple": "#CC79A7", "grey": "#999999",
}

# 示意图块/高亮区的极浅底色
TINT = {
    "blue": "#eaf3fb", "green": "#e8f6ef", "orange": "#fdf3e7",
    "red": "#fcebeb", "grey": "#f4f4f4",
}

# 六方法统一样式：键 = results npz 的 config["method"]（唯一事实来源）。
# band/strict 同蓝色族，实线/虚线 + 不同标记双编码；nominal 灰。
METHOD = {
    "nominal_kcea":       {"color": OI["grey"],       "label": "Nominal KCEA",        "marker": "o", "ls": "-"},
    "pessimistic_resolve": {"color": OI["vermillion"], "label": "Pessimistic re-solve", "marker": "^", "ls": "-"},
    "nsga2":              {"color": OI["green"],      "label": "NSGA-II",             "marker": "v", "ls": "-"},
    "sampled_worst_case": {"color": OI["purple"],     "label": "Sampled worst-case",  "marker": "P", "ls": "-"},
    "bcrea_band":         {"color": OI["blue"],       "label": "BCREA (band)",        "marker": "s", "ls": "-"},
    "bcrea_strict":       {"color": OI["blue"],       "label": "BCREA (strict)",      "marker": "D", "ls": "--"},
}
METHOD_ORDER = ["nominal_kcea", "pessimistic_resolve", "nsga2",
                "sampled_worst_case", "bcrea_band", "bcrea_strict"]

# 规模→颜色/标记（跨规模对比图共用；蓝系渐深 + 黑表示最大规模）
SCALE = {
    24:  {"color": OI["sky"],   "marker": "^", "label": "24 targets"},
    48:  {"color": OI["blue"],  "marker": "s", "label": "48 targets"},
    100: {"color": "#000000",   "marker": "o", "label": "100 targets"},
}

# 线宽层级
LW_DATA = 1.2
LW_EMPH = 1.8
LW_REF = 0.8


def apply():
    """应用论文统一样式 v3。必须在 import matplotlib.pyplot 之前调用。"""
    mpl.rcdefaults()
    mpl.rcParams.update({
        # 字体：Times 族（缺则回退），数学 stix 对齐 IEEEtran
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,      # 项目地板 8pt，禁止更低
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        # 线与坐标轴
        "lines.linewidth": LW_DATA,
        "lines.markersize": 3.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # 去上/右边框；仅 y 向细网格
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        # 图例无框
        "legend.frameon": False,
        # 导出
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,        # TrueType 嵌入，避免 IEEE 字体警告
    })


def panel_tag(ax, letter, dx=-0.02, dy=1.04):
    """面板标签：加粗 10pt 置于轴区左上外侧（全篇统一位置）。"""
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="right")


def require_batch(name=BATCH_V3):
    """核对正式批次存在；不存在立即报错，禁止读旧批次冒充正式结果。"""
    d = RESULTS / name
    if not d.is_dir():
        raise SystemExit(
            f"[style] 正式批次缺失: {d}\n"
            f"性能图只读本轮锁定批次（STYLE.md 第七节）；请先完成正式实验再画此图。")
    return d


def load_io():
    """按文件路径加载同目录 io.py 并缓存为模块 figure_io。

    文件名与标准库 io 同名，而 io 在解释器启动时就在 sys.modules 里，
    普通 `import io` 永远拿到标准库，所以公共数据层必须这样加载：
        import style
        io = style.load_io()
        pool = io.load_pool("nominal_kcea", "gen100_conc_01", 1)
    """
    name = "figure_io"
    if name in sys.modules:
        return sys.modules[name]
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("io.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save(fig, name):
    """最终 PDF 进 paper/figures/，PNG 预览进 figures/build/。"""
    QA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.pdf")
    fig.savefig(QA_DIR / f"{name}.png")
    print(f"[style] saved {OUT_DIR / name}.pdf | {QA_DIR / name}.png")
