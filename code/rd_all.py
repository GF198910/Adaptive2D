"""
率失真对比（全部 9 个数据集）：回应"只在相同 max-error 下比较不够公平"的审稿风险。

方法（ε = rel_tol * std，三者均保证 max|error| <= ε）：
  - daspack_2eps : DASPack 全局均匀量化 step=2ε（论文基线）
  - daspack_1eps : DASPack 全局均匀量化 step=ε（max error=ε/2，
                   把比特均匀撒到全场的替代方案，作为 rate 参照）
  - adaptive_2d  : 论文方法（percentile p=0.2, 块 250x500, 噪声阈值化）

关键分析：对每个数据集，将 adaptive 的工作点 (CR_a, MSE_a) 与
daspack_2eps 的率失真曲线在 CR_a 处对数插值，得到"同码率"下的 MSE 对比。

用法:
  /home/xdzhang/anaconda3/bin/python /home/xdzhang/DAS_Compression/GJI/code/rd_all.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "daspack_src"))
sys.path.insert(0, _HERE)

from daspack import DASCoder, Quantizer            # noqa: E402
from adaptive_daspack_2d import AdaptiveDASPack2D  # noqa: E402

_DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "data"))
_RESULTS_DIR = os.path.normpath(os.path.join(_HERE, "..", "results"))

DATASETS = [
    "7M_S_20231005-214004_RAW", "ZI_C_20230609-102630_RAW", "9G_P", "athens",
    "event_0322_a", "event_0322_b", "event_0620_a", "event_0620_b", "event_0620_c",
]

REL_TOLS = [0.0005, 0.001, 0.002, 0.005, 0.01]
BLOCK_CH, BLOCK_T = 250, 500
FRAC = 0.2
THREADS = 4


def load(name):
    with h5py.File(os.path.join(_DATA_DIR, f"{name}.h5"), "r") as f:
        return np.array(f["data"][:]).astype(np.float32)


import h5py  # noqa: E402


def run_daspack(arr, step, threads=THREADS):
    coder = DASCoder(threads=threads)
    buf = coder.encode(arr.astype(np.float64), Quantizer.Uniform(step=step))
    rec = coder.decode(buf)
    return len(buf), rec


def run_adaptive(arr, eps, threads=THREADS):
    codec = AdaptiveDASPack2D(
        tolerance=eps, block_channels=BLOCK_CH, block_samples=BLOCK_T,
        threshold_mode="percentile", high_energy_fraction=FRAC,
        noise_thresholding=True, threads=threads)
    buf = codec.compress(arr)
    rec = codec.decompress(buf)
    return len(buf), rec


def region_metrics(orig, rec, block_map):
    d = rec.astype(np.float64) - orig.astype(np.float64)
    out = {"max_abs_err": float(np.max(np.abs(d))),
           "mse": float(np.mean(d ** 2))}
    hi, lo = block_map == 1, block_map == 0
    out["mse_high"] = float(np.mean(d[hi] ** 2)) if hi.any() else 0.0
    out["mse_low"] = float(np.mean(d[lo] ** 2)) if lo.any() else 0.0
    return out


def main():
    rows = []
    for name in DATASETS:
        arr = load(name)
        std = float(np.std(arr))
        orig_bytes = arr.nbytes
        print(f"\n>>> {name} shape={arr.shape} std={std:.3f}")

        # percentile 块分类图（区域指标用，与方法无关地固定下来）
        probe = AdaptiveDASPack2D(tolerance=0.1, block_channels=BLOCK_CH,
                                  block_samples=BLOCK_T,
                                  threshold_mode="percentile",
                                  high_energy_fraction=FRAC)
        block_map, _ = probe.get_block_map(arr)

        for rel in REL_TOLS:
            eps = rel * std
            for method in ["daspack_2eps", "daspack_1eps", "adaptive_2d"]:
                t0 = time.perf_counter()
                if method == "daspack_2eps":
                    nbytes, rec = run_daspack(arr, 2.0 * eps)
                elif method == "daspack_1eps":
                    nbytes, rec = run_daspack(arr, 1.0 * eps)
                else:
                    nbytes, rec = run_adaptive(arr, eps)
                dt = time.perf_counter() - t0
                m = region_metrics(arr, rec, block_map)
                cr = orig_bytes / nbytes
                rows.append({"dataset": name, "method": method, "rel_tol": rel,
                             "eps": eps, "cr": cr, "enc_time_s": dt, **m})
                print(f"  [{method:12s}] rel={rel:.4f} CR={cr:7.2f} "
                      f"mse={m['mse']:.3e} mse_high={m['mse_high']:.3e} ({dt:.1f}s)")

    df = pd.DataFrame(rows)
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    out = os.path.join(_RESULTS_DIR, "rd_all_results.csv")
    df.to_csv(out, index=False)
    print(f"\n已保存 {out}")

    # ---- 同码率（rate-matched）分析：adaptive 工作点 vs daspack_2eps 曲线插值 ----
    print("\n" + "=" * 100)
    print("同码率分析：在 adaptive 的 CR 处对 daspack_2eps 的 log-MSE 线性插值")
    print("ratio < 1 表示 adaptive 在相同压缩率下 MSE 更低")
    print("=" * 100)
    print(f"{'dataset':>26} {'rel':>7} {'CR_a':>7} {'CR_dp*':>7} "
          f"{'MSE ratio':>10} {'MSE_hi ratio':>12}")
    summary = []
    for name in DATASETS:
        dp = df[(df.dataset == name) & (df.method == "daspack_2eps")].sort_values("cr")
        ad = df[(df.dataset == name) & (df.method == "adaptive_2d")]
        if len(dp) < 2:
            continue
        lcr = np.log(dp["cr"].values)
        for _, r in ad.iterrows():
            tgt = np.log(r["cr"])
            if tgt < lcr.min() or tgt > lcr.max():
                continue
            mse_dp = np.exp(np.interp(tgt, lcr, np.log(dp["mse"].values)))
            mseh_dp = np.exp(np.interp(tgt, lcr, np.log(dp["mse_high"].values)))
            ratio = r["mse"] / mse_dp
            ratio_h = r["mse_high"] / mseh_dp
            summary.append({"dataset": name, "rel_tol": r["rel_tol"],
                            "mse_ratio_at_matched_cr": ratio,
                            "mse_high_ratio_at_matched_cr": ratio_h,
                            "cr_adaptive": r["cr"]})
            print(f"{name:>26} {r['rel_tol']:>7.4f} {r['cr']:>7.2f} "
                  f"{np.exp(tgt):>7.2f} {ratio:>10.3f} {ratio_h:>12.3f}")
    sdf = pd.DataFrame(summary)
    sdf.to_csv(os.path.join(_RESULTS_DIR, "rd_all_rate_matched.csv"), index=False)
    if len(sdf):
        print("-" * 100)
        print(f"{'MEAN':>26} {'':>7} {'':>7} {'':>7} "
              f"{sdf['mse_ratio_at_matched_cr'].mean():>10.3f} "
              f"{sdf['mse_high_ratio_at_matched_cr'].mean():>12.3f}")


if __name__ == "__main__":
    main()
