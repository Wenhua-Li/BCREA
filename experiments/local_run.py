"""本地并行任务执行器（2026-10-08 起替代原服务器流水线，规矩见 AGENTS.md 第 7 条）。

用法（在仓库根目录执行）：
    python experiments/local_run.py --taskfile data/tasks/日期_标签.txt \
        --out data/results/日期_方向_参数标签 [-j 20]

任务清单：每行一条 shell 命令，# 开头是注释，空行跳过；命令里用 $PYTHON 代替
python（自动替换成探测出的实验解释器，见下），路径写相对仓库根的路径。与原服务
器流水线同格式，历史任务清单不用改。

解释器选择：$PYTHON 替换成哪个 python 由三步定——①环境变量 WTA_PYTHON（想固定
就设它）；②自动探测：依次试 sys.executable 和 PATH 里的 python，用第一个能
import numpy+scipy 的（避开缺科学计算栈的捆绑 python，如 Inkscape 的）；③都失败
就退回 sys.executable 并提示。

输出（落在 --out 目录，与原流水线同结构）：
    logs/job_00001.log   每个任务自己的 stdout/stderr（编号=在清单里的第几条有效命令）
    summary.txt          逐条 OK/FAIL + 首尾统计行
看进度直接看 summary.txt，看单个任务输出看 logs/job_*.log——文件都在本地，无需额外命令。

纪律（同原流水线）：输出目录已有内容时拒绝执行（防覆盖旧结果）；确认续跑/重跑加
--force。并行的是任务个数（进程级），默认 20；单任务内部 BLAS 线程限成 1，并行度
靠任务数不靠线程数。命令含 --backend gpu/auto 时受 --gpu-jobs 限制，默认同一张卡只跑 1 个。
"""
from __future__ import annotations

import argparse
import os
import shutil
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# 本执行器常以无控制台方式（DETACHED_PROCESS）运行；Windows 上再起控制台子进程时
# 不带 CREATE_NO_WINDOW 会各弹一个黑框，故所有 subprocess 启动都带上它（非 Windows 为 0）。
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

REPO = Path(__file__).resolve().parents[1]
DEFAULT_JOBS = 20   # 本机口径：默认 20 个任务并行（2026-10-08 用户拍板）

_write_lock = threading.Lock()   # 多线程追加 summary.txt 用


def _all_pythons() -> list[str]:
    """PATH 里全部 python 候选（where/which -a 可能给出多个，顺序即优先序）。"""
    finder = ["where", "python"] if os.name == "nt" else ["which", "-a", "python", "python3"]
    try:
        r = subprocess.run(finder, capture_output=True, text=True, timeout=15,
                           creationflags=_NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [ln.strip('"') for ln in r.stdout.splitlines() if ln.strip()]


def pick_python() -> str:
    """定 $PYTHON 用哪个解释器：WTA_PYTHON > 第一个带 numpy+scipy 的 > sys.executable。"""
    candidates: list[str] = []
    if os.environ.get("WTA_PYTHON"):
        candidates.append(os.environ["WTA_PYTHON"])
    if sys.executable:
        candidates.append(sys.executable)
    for cand in _all_pythons():
        if cand not in candidates:
            candidates.append(cand)
    for cand in candidates:
        try:
            r = subprocess.run([cand, "-c", "import numpy, scipy"],
                               capture_output=True, timeout=60,
                               creationflags=_NO_WINDOW)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode == 0:
            return cand
    print(f"[提示] 没找到能 import numpy+scipy 的 python，退回 {sys.executable}")
    return sys.executable


def run_tasks(taskfile: Path, outdir: Path, jobs: int, force: bool, gpu_jobs: int,
              append_args: str = "") -> int:
    if outdir.exists() and any(outdir.iterdir()) and not force:
        sys.exit(
            f"{outdir} 已有内容。按'不许覆盖旧结果'的规矩请换输出目录；"
            "确属续跑/重跑再加 --force。"
        )
    if gpu_jobs < 1:
        sys.exit("--gpu-jobs 必须至少为 1")
    py = pick_python()
    lines = taskfile.read_text(encoding="utf-8-sig").splitlines()
    raw = [ln.rstrip("\r").strip() for ln in lines]
    raw = [ln for ln in raw if ln and not ln.startswith("#")]
    if not raw:
        sys.exit(f"任务清单里没有有效命令：{taskfile}")
    # 文本替换（带引号），不靠 env 展开——解释器路径可能含空格（如 C:/Program Files/...）
    cmds = [ln.replace("$PYTHON", f'"{py}"') for ln in raw]
    extra = append_args.strip()
    if extra:   # 追加参数（如 "--backend gpu"）：冻结的任务清单原地跑 GPU，不必重生成清单
        cmds = [f"{c} {extra}" for c in cmds]

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "logs").mkdir(exist_ok=True)
    summary = outdir / "summary.txt"

    env = dict(os.environ)
    # 单任务只用 1 个 BLAS 线程，并行度全靠任务数
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = "1"

    bash = shutil.which("bash") or "bash"
    gpu_slots = threading.Semaphore(gpu_jobs)
    with summary.open("a", encoding="utf-8") as sf:
        sf.write(f"开始: 本地跑  任务文件={taskfile.as_posix()}  输出={outdir.as_posix()}  "
                 f"并行={jobs}  GPU并行={gpu_jobs}  任务数={len(cmds)}"
                 + (f"  追加参数={extra}" if extra else "") + f"\n解释器: {py}\n")

    def one(idx: int, cmd: str, display: str) -> None:
        log = outdir / "logs" / f"job_{idx:05d}.log"
        uses_gpu = bool(re.search(r"(?:^|\\s)--backend(?:=|\\s+)(?:gpu|auto)(?:\\s|$)", cmd))
        if uses_gpu:
            gpu_slots.acquire()
        try:
            try:
                r = subprocess.run([bash, "-c", cmd], cwd=REPO, env=env,
                                   capture_output=True, text=True, errors="replace",
                                   creationflags=_NO_WINDOW)
                log.write_text(r.stdout + r.stderr, encoding="utf-8")
                mark = "OK  " if r.returncode == 0 else f"FAIL rc={r.returncode}"
                with _write_lock, summary.open("a", encoding="utf-8") as sf:
                    sf.write(f"{mark} {idx:05d}  {display}\n")
            except Exception as exc:
                # 子进程没起来（资源不足等）也要留痕：异常进日志、summary 记 FAIL，
                # 否则线程池会静默吞掉异常，任务就在 summary 里凭空消失
                msg = f"{type(exc).__name__}: {exc}".replace("\r", " ").replace("\n", " ")
                mark = f"FAIL exc={msg[:200]}"
                with log.open("a", encoding="utf-8") as lf:
                    lf.write(f"FAIL exc={msg}\n")
                with _write_lock, summary.open("a", encoding="utf-8") as sf:
                    sf.write(f"{mark} {idx:05d}  {display}\n")
            print(f"{mark} {idx:05d}  {display}", flush=True)
        finally:
            if uses_gpu:
                gpu_slots.release()

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for i, (cmd, display) in enumerate(zip(cmds, raw), start=1):
            pool.submit(one, i, cmd, display)
    # ThreadPoolExecutor 的 with 块退出时已等待全部任务结束

    text = summary.read_text(encoding="utf-8")
    ok = sum(1 for ln in text.splitlines() if ln.startswith("OK"))
    fail = sum(1 for ln in text.splitlines() if ln.startswith("FAIL"))
    print(f"结束: 共 {len(cmds)} 个任务: OK={ok} FAIL={fail}  明细见 {summary}")
    with summary.open("a", encoding="utf-8") as sf:
        sf.write(f"结束: 共 {len(cmds)} 个任务: OK={ok} FAIL={fail}\n")
    return 0 if fail == 0 else 1


def main() -> None:
    ap = argparse.ArgumentParser(
        description="本地并行任务执行器（任务清单格式与旧服务器流水线相同）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--taskfile", required=True, help="任务清单文件（每行一条命令，相对仓库根）")
    ap.add_argument("--out", required=True, help="输出目录（相对仓库根，如 data/results/日期_方向_标签）")
    ap.add_argument("-j", "--jobs", type=int, default=DEFAULT_JOBS,
                    help=f"并行任务数，默认 {DEFAULT_JOBS}")
    ap.add_argument("--force", action="store_true",
                    help="输出目录已存在时允许续跑（默认拒绝，防覆盖旧结果）")
    ap.add_argument("--gpu-jobs", type=int, default=1,
                    help="同一张 GPU 同时运行的任务数，默认 1；仅限制命令含 --backend gpu/auto 的任务")
    ap.add_argument("--append-args", default="",
                    help='给每条任务命令追加的参数（如 --append-args "--backend gpu"），不改原任务清单')
    args = ap.parse_args()

    taskfile = Path(args.taskfile)
    if not taskfile.is_absolute():
        taskfile = REPO / taskfile
    if not taskfile.exists():
        sys.exit(f"找不到任务清单文件：{args.taskfile}")
    outdir = Path(args.out.strip("/"))
    if outdir.is_absolute():
        sys.exit("--out 请写相对仓库根的路径（保证结果落在 data/results/ 的存档纪律）")

    sys.exit(run_tasks(taskfile, outdir, args.jobs, args.force, args.gpu_jobs, args.append_args))


if __name__ == "__main__":
    main()
