"""GPU 批量解码后端（单融合 CUDA kernel，CuPy RawKernel）。

用途：把 core.decoder.evaluate 的逐个体 Python 解码整体搬到 GPU——一个线程
解码一个个体，发射轮次/目标顺序仍在核内串行，个体维完全并行。一次 kernel 发射
处理整池，消除"逐目标·逐组合"的小 kernel 发射开销（那是上一版 CuPy 实现的瓶颈）。

语义对齐（与 decoder.py 的 CPU 实现同口径，逐项对照移植）：
  - 弹药限制 → 概率表查询 → 轮间时间窗行扫描（含 t_cap=目标飞行秒数上限）→
    preT/换雷达约束 → 雷达通道占用 → 失败传播 fail[t, k:]；
    infeasible_ok=True 的"表缺失按失败处理"是默认行为；
  - 通道占用不存稠密 colsum（nS≈1001，逐线程存不下），改为每个体记录已接任务
    （雷达、窗口、need、可计入槽位掩码）与自身窗口，按需汇总出与 CPU colsum
    等价的逐槽占用。注意：CPU 的 occupied 用"全局槽下标"去切局部数组
    （s0>1 时起点落在数组界外，等价于几乎不标记），本内核按 Python 切片的
    截断语义原样复刻——这是既有实验结果的真实口径，不得"顺手修正"
    （修了会改变历史数字；要改必须先改 decoder.py 并重跑全部实验）；
  - kdg 记录点与 CPU 三处 append（越界断扫 / preT 换雷达 / 失败收尾）逐一对应；
  - 目标值不在核内求和：kernel 只回传 kill_prob (B,N,K)，Python 侧用与
    decoder.py 相同的 numpy 表达式算 survive 与 obj → 数值逐位一致；
  - infeasible_ok=False 时 CPU 抛 ValueError：kernel 记录每个体首个"表缺失"事件，
    Python 侧按个体序首犯抛同款错误。

结果口径：objs / kill_prob / fail / kdg / tasks 与 CPU evaluate 逐位相同（实测验收，
见 tests/test_gpu_backend.py）；GPU 不可用时由调用方回退 CPU。
"""
from __future__ import annotations

import math

import numpy as np

from core.decoder import EvalResult, Task, evaluate

_CUPY = None
_CUPY_ERROR: str | None = None

# N 的分桶（核内本地数组按桶尺寸静态分配；桶越贴合 N，本地内存越省）
_BUCKETS = (12, 16, 24, 32, 48, 64, 100, 128, 160, 200, 256, 384, 512, 768)

_KERNELS: dict[tuple, object] = {}

_SRC_TEMPLATE = r"""
#define NB @NB@
#define KB @KB@
#define JB @JB@
#define RB @RB@
#define TCAP @TCAP@
#define KCAP @KCAP@
#define WMAX @WMAX@

__device__ __forceinline__ double d_nan() { return __longlong_as_double(0x7ff8000000000000ULL); }

extern "C" __global__ void swta_decode(
    const int* __restrict__ decs,              /* (K,N,2,B) 连续布局 */
    const int B,
    const int N, const int K, const int J, const int R, const int nS,
    const int* __restrict__ order,
    const int* __restrict__ tcap,
    const int* __restrict__ load_lim,          /* (J) */
    const int* __restrict__ cap_r,             /* (R) */
    const int* __restrict__ need_rt,           /* (R,N) */
    const int* __restrict__ offs,              /* (N,J,R) */
    const int* __restrict__ tlens,             /* (N,J,R) */
    const double* __restrict__ tab_p,          /* 扁平行 [P] */
    const double* __restrict__ tab_tl,         /* 扁平行 [发射时刻] */
    const double* __restrict__ tab_tk,         /* 扁平行 [命中时刻] */
    const double d1lo, const double d1hi, const double preT, const double dT,
    double* __restrict__ kp_out,               /* (B,N,K) */
    unsigned char* __restrict__ fail_out,      /* (B,N,K) */
    int* __restrict__ kdg_out,                 /* (B,KCAP) 打包 k*NB+pos */
    int* __restrict__ kdg_cnt,                 /* (B) */
    int* __restrict__ task_cnt,                /* (B) */
    short* __restrict__ task_t,                /* (B,TCAP) */
    unsigned char* __restrict__ task_k,        /* (B,TCAP) */
    short* __restrict__ task_row,              /* (B,TCAP) */
    int* __restrict__ bad_cnt,                 /* (B) 表缺失事件数（infeasible_ok=False 用） */
    short* __restrict__ bad_t,                 /* (B) 首个事件 */
    unsigned char* __restrict__ bad_k,         /* (B) */
    unsigned char* __restrict__ bad_j,         /* (B) */
    unsigned char* __restrict__ bad_r)         /* (B) */
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    short fail_from[NB];
    double lastk[NB];
    unsigned char lastr[NB];
    int lu[JB];

    /* 已接任务记录（按雷达拉链；通道占用检查按需汇总） */
    short t_t[TCAP];
    unsigned char t_k[TCAP];
    short t_row[TCAP];
    short t_s0[TCAP];
    short t_s1[TCAP];
    unsigned char t_r[TCAP];
    unsigned char t_need[TCAP];
    unsigned int t_mask[TCAP];   /* 该任务可计入占用的槽位掩码（局部下标位） */
    short t_next[TCAP];
    short head[RB];

    /* 每目标自身已 append 的窗口（同雷达复用时按 CPU 语义标记 occupied） */
    unsigned char wc[NB];
    unsigned char w_r[NB * KB];
    short w_s0[NB * KB];
    short w_s1[NB * KB];

    for (int i = 0; i < N; ++i) {
        fail_from[i] = (short)K;
        lastk[i] = d_nan();
        lastr[i] = 255;
        wc[i] = 0;
    }
    for (int i = 0; i < J; ++i) lu[i] = 0;
    for (int i = 0; i < R; ++i) head[i] = -1;

    int nt = 0, nk = 0, nbad = 0;

    for (int k = 0; k < K; ++k) {
        for (int pos = 0; pos < N; ++pos) {
            const short t = (short)order[pos];
            if (k >= (int)fail_from[t]) continue;
            const int lb = ((k * N + t) * 2) * B + b;
            const int j = decs[lb];
            const int r = decs[lb + B];
            if (j < 0 || j >= J || r < 0 || r >= R) { fail_from[t] = (short)k; continue; }
            if (lu[j] >= load_lim[j]) { fail_from[t] = (short)k; continue; }
            const int tb = ((int)t * J + j) * R + r;
            const int len = tlens[tb];
            if (len <= 0) {
                if (nbad == 0) {
                    bad_t[b] = t; bad_k[b] = (unsigned char)k;
                    bad_j[b] = (unsigned char)j; bad_r[b] = (unsigned char)r;
                }
                ++nbad;
                fail_from[t] = (short)k;
                continue;
            }
            const int off0 = offs[tb];
            int row = 0;
            if (k > 0) {
                const double last = lastk[t];
                double delta = tab_tk[off0] - last;
                int ok = (delta >= d1lo) && (delta <= d1hi) ? 1 : 0;
                while (!ok) {
                    ++row;
                    if (row >= len || row > tcap[t]) {
                        if (nk < KCAP) kdg_out[(long)b * KCAP + nk] = k * NB + pos;
                        ++nk;
                        break;
                    }
                    delta = tab_tk[off0 + row] - last;
                    ok = (delta >= d1lo) && (delta <= d1hi) ? 1 : 0;
                }
                if (delta < preT && (int)lastr[t] != r) {
                    if (nk < KCAP) kdg_out[(long)b * KCAP + nk] = k * NB + pos;
                    ++nk;
                    ok = 0;
                }
                if (!ok) {
                    fail_from[t] = (short)k;
                    if (nk < KCAP) kdg_out[(long)b * KCAP + nk] = k * NB + pos;
                    ++nk;
                    continue;
                }
            }
            const double tk = tab_tk[off0 + row];
            int s0 = (int)floor((tk - preT) / dT + 0.5);
            if (s0 < 1) s0 = 1;
            int s1 = (int)floor(tk / dT + 0.5);
            if (s1 > nS) s1 = nS;

            const int need = need_rt[r * N + t];
            int conflict = 0;
            unsigned int new_mask = 0u;
            const int wlen = (s1 >= s0) ? (s1 - s0 + 1) : 0;
            if (wlen > 0) {
                unsigned char occ_local[WMAX];
                short load[WMAX];
                for (int i = 0; i < wlen; ++i) { occ_local[i] = 0; load[i] = 0; }
                /* 自身窗口：复刻 CPU occupied[max(a,s0)-1 : min(b,s1)] 的原始下标切片
                   （s0>1 时起点越界 → Python 切片截断为不标记；这里原样复现） */
                for (int wj = 0; wj < (int)wc[t]; ++wj) {
                    const int idx = (int)t * KB + wj;
                    if ((int)w_r[idx] != r) continue;
                    int lo = ((int)w_s0[idx] > s0 ? (int)w_s0[idx] : s0) - 1;
                    int hi = ((int)w_s1[idx] < s1 ? (int)w_s1[idx] : s1);
                    if (lo < 0) lo = 0;
                    if (hi < 0) hi = 0;
                    if (lo > wlen) lo = wlen;
                    if (hi > wlen) hi = wlen;
                    for (int j2 = lo; j2 < hi; ++j2) occ_local[j2] = 1;
                }
                /* colsum 等价量：按任务可计入槽位（掩码）累加（同目标多任务重复计入） */
                for (int q = head[r]; q >= 0; q = t_next[q]) {
                    const int qs0 = (int)t_s0[q];
                    const unsigned int qmask = t_mask[q];
                    const int qn = (int)t_need[q];
                    int jlo = qs0 - s0; if (jlo < 0) jlo = 0;
                    int jhi = qs0 - s0 + 31; if (jhi > wlen - 1) jhi = wlen - 1;
                    for (int j2 = jlo; j2 <= jhi; ++j2) {
                        const int kq = (s0 + j2) - qs0;
                        if ((qmask >> kq) & 1u) load[j2] += (short)qn;
                    }
                }
                const int capv = cap_r[r];
                for (int j2 = 0; j2 < wlen; ++j2) {
                    if (occ_local[j2]) continue;
                    if ((int)load[j2] + need > capv) { conflict = 1; break; }
                }
                if (!conflict) {
                    for (int j2 = 0; j2 < wlen; ++j2)
                        if (!occ_local[j2]) new_mask |= (1u << j2);
                }
            }
            if (conflict) { fail_from[t] = (short)k; continue; }

            const double pk = tab_p[off0 + row];
            lu[j] += 1;
            lastk[t] = tk;
            lastr[t] = (unsigned char)r;
            kp_out[((long)b * N + t) * K + k] = pk;
            if (nt < TCAP) {
                t_t[nt] = t; t_k[nt] = (unsigned char)k; t_row[nt] = (short)row;
                t_s0[nt] = (short)s0; t_s1[nt] = (short)s1; t_r[nt] = (unsigned char)r;
                t_need[nt] = (unsigned char)need; t_mask[nt] = new_mask;
                t_next[nt] = head[r];
                head[r] = (short)nt;
                ++nt;
            }
            if (new_mask != 0u && (int)wc[t] < KB) {
                const int idx = (int)t * KB + (int)wc[t];
                w_r[idx] = (unsigned char)r; w_s0[idx] = (short)s0; w_s1[idx] = (short)s1;
                ++wc[t];
            }
        }
    }

    for (int t = 0; t < N; ++t)
        for (int k = 0; k < K; ++k)
            fail_out[((long)b * N + t) * K + k] = (k >= (int)fail_from[t]) ? (unsigned char)1 : (unsigned char)0;

    kdg_cnt[b] = nk;
    task_cnt[b] = nt;
    bad_cnt[b] = nbad;
    for (int q = 0; q < nt; ++q) {
        task_t[(long)b * TCAP + q] = t_t[q];
        task_k[(long)b * TCAP + q] = t_k[q];
        task_row[(long)b * TCAP + q] = t_row[q];
    }
}
"""


def gpu_info() -> dict:
    """返回 GPU backend 状态，供 CLI 结果 config 记录。"""
    global _CUPY, _CUPY_ERROR
    if _CUPY is None and _CUPY_ERROR is None:
        try:
            import cupy as cp
            if int(cp.cuda.runtime.getDeviceCount()) <= 0:
                raise RuntimeError("未检测到 CUDA 设备")
            probe = cp.zeros(1, dtype=cp.float32)
            probe += cp.float32(1)
            cp.cuda.Stream.null.synchronize()
            del probe
            _CUPY = cp
        except Exception as exc:  # CuPy 缺失或 CUDA 不可用
            _CUPY_ERROR = f"{type(exc).__name__}: {exc}"
    if _CUPY is None:
        return {"available": False, "backend": "cpu", "error": _CUPY_ERROR}
    dev = _CUPY.cuda.Device().id
    return {
        "available": True,
        "backend": "cupy-kernel",
        "device_count": int(_CUPY.cuda.runtime.getDeviceCount()),
        "device": dev,
        "name": _CUPY.cuda.runtime.getDeviceProperties(dev)["name"].decode(errors="replace"),
        "cupy": _CUPY.__version__,
    }


def resolve_backend(backend: str = "cpu") -> str:
    """解析 cpu/gpu/auto；显式 gpu 不可用时抛错，auto 不可用时回退 cpu。"""
    name = str(backend).lower()
    if name not in {"cpu", "gpu", "auto"}:
        raise ValueError(f"未知 backend={backend!r}，应为 cpu、gpu 或 auto")
    if name == "cpu":
        return "cpu"
    info = gpu_info()
    if info["available"]:
        return "gpu"
    if name == "auto":
        return "cpu"
    raise RuntimeError(f"backend='gpu' 不可用：{info.get('error', '无 CUDA 设备')}")


def _bucket_for(n: int) -> int:
    for b in _BUCKETS:
        if n <= b:
            return b
    raise ValueError(f"N={n} 超过 GPU 后端最大分桶 {_BUCKETS[-1]}，请扩展 _BUCKETS")


def _kernel_for(nb: int, kb: int, jb: int, rb: int, wmax: int):
    key = (nb, kb, jb, rb, wmax)
    if key not in _KERNELS:
        src = (
            _SRC_TEMPLATE
            .replace("@NB@", str(nb))
            .replace("@KB@", str(kb))
            .replace("@JB@", str(jb))
            .replace("@RB@", str(rb))
            .replace("@TCAP@", str(kb * nb))
            .replace("@KCAP@", str(3 * kb * nb))
            .replace("@WMAX@", str(wmax))
        )
        _KERNELS[key] = _CUPY.RawKernel(src, "swta_decode")
    return _KERNELS[key]


def _problem_tables(problem) -> dict:
    """把 SWTAProblem 的概率表/参数摊平成定长张量（CPU numpy，随后按需上卡）。

    布局：
      offs/tlens: (N,J,R) int32——每个 (目标, 发射车, 雷达) 的表在扁平数组里的起点与行数；
      tab_p/tab_tl/tab_tk: 扁平行 [P, 发射时刻, 命中时刻]。
    其余为按部/按类型的参数展开（弹药上限、通道上限、通道占用系数）。
    """
    cached = getattr(problem, "_gpu_tables", None)
    if cached is not None:
        return cached

    p = problem.params
    N, J, R = problem.n_target, p.n_missile, p.n_radar
    if J > 255 or R > 255 or int(problem.max_missiles) > 255:
        raise ValueError("GPU 后端要求 J/R/K ≤ 255（当前部署超出）")
    if int(problem.nS) > 32767:
        raise ValueError("GPU 后端要求 nS ≤ 32767")
    wmax = int(math.ceil(p.preT / p.dT)) + 8
    if wmax > 32:
        raise ValueError(f"GPU 后端要求 preT/dT+8 ≤ 32（当前 {wmax}）：窗口掩码用 32 位")

    offs = np.zeros((N, J, R), dtype=np.int32)
    tlens = np.zeros((N, J, R), dtype=np.int32)
    rows: list[np.ndarray] = []
    total = 0
    for t, tgt in enumerate(problem.targets):
        for (j, r), tbl in tgt.pkill.items():
            n = len(tbl)
            if n <= 0:
                continue
            offs[t, j, r] = total
            tlens[t, j, r] = n
            rows.append(np.asarray(tbl, dtype=np.float64))
            total += n
    if rows:
        all_rows = np.concatenate(rows, axis=0)
        tab_p = np.ascontiguousarray(all_rows[:, 0])
        tab_tl = np.ascontiguousarray(all_rows[:, 1])
        tab_tk = np.ascontiguousarray(all_rows[:, 2])
    else:
        tab_p = tab_tl = tab_tk = np.zeros(0, dtype=np.float64)

    rtype = np.asarray(p.radar_type) - 1
    need_rt = np.ascontiguousarray(
        np.asarray(p.radar_coefficient)[rtype][:, np.asarray(problem.ttype) - 1], dtype=np.int32)
    if need_rt.size and int(need_rt.max()) > 255:
        raise ValueError("GPU 后端要求通道占用系数 ≤ 255")

    data = {
        "N": N, "J": J, "R": R, "K": int(problem.max_missiles),
        "nS": int(problem.nS),
        "order": np.ascontiguousarray(problem.target_ind, dtype=np.int32),
        "tcap": np.ascontiguousarray([tgt.T for tgt in problem.targets], dtype=np.int32),
        "load_lim": np.ascontiguousarray(
            np.asarray(p.missile_load_limit)[np.asarray(p.missile_type) - 1], dtype=np.int32),
        "cap_r": np.ascontiguousarray(np.asarray(p.radar_n_channel)[rtype], dtype=np.int32),
        "need_rt": need_rt,
        "offs": offs, "tlens": tlens,
        "tab_p": tab_p, "tab_tl": tab_tl, "tab_tk": tab_tk,
        "d1lo": float(p.delta_t1[0]), "d1hi": float(p.delta_t1[1]),
        "preT": float(p.preT), "dT": float(p.dT),
        "nb": _bucket_for(N),
        "wmax": wmax,
    }
    problem._gpu_tables = data
    return data


def _device_tables(problem, data: dict) -> dict:
    cached = data.get("_dev")
    if cached is None:
        cp = _CUPY
        cached = {k: cp.asarray(data[k]) for k in
                  ("order", "tcap", "load_lim", "cap_r", "need_rt", "offs", "tlens",
                   "tab_p", "tab_tl", "tab_tk")}
        data["_dev"] = cached
    return cached


def _launch(problem, decs_pool: np.ndarray, chunk: int, needed=("kp",)):
    """跑一次 kernel，按 needed 只回传要用的 host 侧输出（省设备→主机拷贝与 host 处理）。

    needed 取值：kp（命中概率矩阵）、fail、kdg、tasks、bad（表缺失事件，抛错检查用）；
    设备端数组一律照常申请（kernel 无条件写入），只是没要的就不拷回。
    """
    cp = _CUPY
    data = _problem_tables(problem)
    dev = _device_tables(problem, data)
    N, K = data["N"], data["K"]
    nb = data["nb"]
    tcap_sz = K * nb
    kcap_sz = 3 * K * nb
    kernel = _kernel_for(nb, K, data["J"], data["R"], data["wmax"])

    decs = np.asarray(decs_pool)
    B = len(decs)
    kp = cp.zeros((B, N, K), dtype=cp.float64)
    fail = cp.empty((B, N, K), dtype=cp.uint8)
    kdg_out = cp.empty((B, kcap_sz), dtype=cp.int32)
    kdg_cnt = cp.empty(B, dtype=cp.int32)
    task_cnt = cp.empty(B, dtype=np.int32)
    task_t = cp.empty((B, tcap_sz), dtype=np.int16)
    task_k = cp.empty((B, tcap_sz), dtype=np.uint8)
    task_row = cp.empty((B, tcap_sz), dtype=np.int16)
    bad_cnt = cp.empty(B, dtype=np.int32)
    bad_t = cp.empty(B, dtype=np.int16)
    bad_k = cp.empty(B, dtype=np.uint8)
    bad_j = cp.empty(B, dtype=np.uint8)
    bad_r = cp.empty(B, dtype=np.uint8)

    d = cp.asarray(decs, dtype=cp.int32)                     # (B,K,N,2)
    d = cp.ascontiguousarray(cp.transpose(d, (1, 2, 3, 0)))  # (K,N,2,B)

    block = 128
    grid = (B + block - 1) // block
    kernel(
        (grid,), (block,),
        (d, np.int32(B), np.int32(N), np.int32(K), np.int32(data["J"]), np.int32(data["R"]),
         np.int32(data["nS"]), dev["order"], dev["tcap"], dev["load_lim"], dev["cap_r"],
         dev["need_rt"], dev["offs"], dev["tlens"], dev["tab_p"], dev["tab_tl"], dev["tab_tk"],
         np.float64(data["d1lo"]), np.float64(data["d1hi"]), np.float64(data["preT"]),
         np.float64(data["dT"]),
         kp, fail, kdg_out, kdg_cnt, task_cnt, task_t, task_k, task_row,
         bad_cnt, bad_t, bad_k, bad_j, bad_r),
    )
    raw: dict = {"nb": nb}
    if "kp" in needed:
        raw["kp"] = cp.asnumpy(kp)
    if "fail" in needed:
        raw["fail"] = cp.asnumpy(fail).astype(bool)
    if "kdg" in needed:
        raw["kdg_cnt"] = cp.asnumpy(kdg_cnt)
        raw["kdg_data"] = cp.asnumpy(kdg_out)
    if "tasks" in needed:
        raw["task_cnt"] = cp.asnumpy(task_cnt)
        raw["task_t"] = cp.asnumpy(task_t)
        raw["task_k"] = cp.asnumpy(task_k)
        raw["task_row"] = cp.asnumpy(task_row)
    if "bad" in needed:
        raw["bad_cnt"] = cp.asnumpy(bad_cnt)
        raw["bad_t"] = cp.asnumpy(bad_t)
        raw["bad_k"] = cp.asnumpy(bad_k)
        raw["bad_j"] = cp.asnumpy(bad_j)
        raw["bad_r"] = cp.asnumpy(bad_r)
    return raw


def _objs_from_kp(kp: np.ndarray) -> np.ndarray:
    """与 decoder.evaluate 完全相同的目标值表达式：1-Π(1-p) 再按目标求和。"""
    survive = 1.0 - np.prod(1.0 - kp, axis=2)
    return survive.sum(axis=1)


def decode_objs(problem, decs_pool, batch_size: int = 256) -> np.ndarray:
    """批量目标值（只用 objs 的路径，如扰动扫描/鲁棒评分）——只拷回 kp 一个数组。"""
    decs = np.asarray(decs_pool)
    B = len(decs)
    if B == 0:
        return np.zeros(0, dtype=np.float64)
    chunk = max(1, int(batch_size)) if batch_size and batch_size > 0 else B
    out = np.empty(B, dtype=np.float64)
    for s in range(0, B, chunk):
        e = min(s + chunk, B)
        raw = _launch(problem, decs[s:e], e - s, ("kp",))
        out[s:e] = _objs_from_kp(raw["kp"])
    return out


def _rebuild(problem, decs: np.ndarray, s: int, e: int, raw: dict, with_tasks: bool) -> list[EvalResult]:
    """host 侧把 kernel 原始输出重建为 EvalResult（objs 用与 CPU 相同的 numpy 表达式）。"""
    data = _problem_tables(problem)
    kp3 = raw["kp"]                                    # (n,N,K)
    survive = 1.0 - np.prod(1.0 - kp3, axis=2)         # (n,N)
    nb = raw["nb"]
    out: list[EvalResult] = []
    for i in range(e - s):
        b = s + i
        cnt_k = int(raw["kdg_cnt"][i])
        packed = raw["kdg_data"][i, :cnt_k]
        pairs = np.empty((cnt_k, 2), dtype=np.int32)
        if cnt_k:
            pairs[:, 0] = packed // nb
            pairs[:, 1] = packed % nb
        tasks: list[Task] = []
        if with_tasks:
            cnt_t = int(raw["task_cnt"][i])
            for q in range(cnt_t):
                t = int(raw["task_t"][i, q])
                k = int(raw["task_k"][i, q])
                row = int(raw["task_row"][i, q])
                j = int(decs[b, k, t, 0])
                r = int(decs[b, k, t, 1])
                off0 = int(data["offs"][t, j, r])
                tasks.append(Task(
                    t, k, j, r, row,
                    float(data["tab_p"][off0 + row]),
                    float(data["tab_tl"][off0 + row]),
                    float(data["tab_tk"][off0 + row]),
                ))
        out.append(EvalResult(
            obj=float(survive[i].sum()),
            kdg=pairs,
            tasks=tasks,
            kill_prob=np.column_stack([kp3[i], survive[i]]),
            fail=raw["fail"][i],
        ))
    return out


def decode_full(problem, decs_pool, batch_size: int = 256) -> list[EvalResult]:
    """完整解码结果（objs/kill_prob/fail/kdg/tasks）。"""
    decs = np.asarray(decs_pool)
    B = len(decs)
    if B == 0:
        return []
    chunk = max(1, int(batch_size)) if batch_size and batch_size > 0 else B
    out: list[EvalResult] = []
    for s in range(0, B, chunk):
        e = min(s + chunk, B)
        raw = _launch(problem, decs[s:e], e - s, ("kp", "fail", "kdg", "tasks"))
        out.extend(_rebuild(problem, decs, s, e, raw, with_tasks=True))
    return out


def evaluate_batch(problem, decs_pool, infeasible_ok: bool = False,
                   backend: str = "cpu", batch_size: int = 256) -> list[EvalResult]:
    """批量评估一个解池；返回 list[EvalResult]（与逐个体 evaluate 同口径）。

    cpu：逐个体调用 decoder.evaluate（与历史代码完全一致）。
    gpu：单融合 kernel + CPU 侧重建；infeasible_ok=False 时若出现表缺失，
         按 CPU 的抛出语义抛 ValueError（消息格式一致）。抛错检查与主解码共用
         同一次 kernel 发射（2026-10-16 修正：原实现二次发射+二次拷贝，白花约 1/4 时间）。
    """
    selected = resolve_backend(backend)
    if selected == "cpu":
        return [evaluate(problem, decs_pool[i], infeasible_ok=infeasible_ok) for i in range(len(decs_pool))]
    decs = np.asarray(decs_pool)
    B = len(decs)
    if B == 0:
        return []
    chunk = max(1, int(batch_size)) if batch_size and batch_size > 0 else B
    out: list[EvalResult] = []
    for s in range(0, B, chunk):
        e = min(s + chunk, B)
        need = ("kp", "fail", "kdg", "tasks", "bad") if not infeasible_ok else ("kp", "fail", "kdg", "tasks")
        raw = _launch(problem, decs[s:e], e - s, need)
        if not infeasible_ok:
            for i in range(e - s):
                if int(raw["bad_cnt"][i]) > 0:
                    t, k, j, r = (int(raw["bad_t"][i]), int(raw["bad_k"][i]),
                                  int(raw["bad_j"][i]), int(raw["bad_r"][i]))
                    raise ValueError(f"目标{t}第{k}发分配了不可行的 (发射车{j}, 雷达{r})")
        out.extend(_rebuild(problem, decs, s, e, raw, with_tasks=True))
    return out
