"""
STA/LTA 事件检测验证：MiDAS 数据集（5 个事件段）。

对每个 MiDAS 数据集运行经典 STA/LTA 检测，对比：
  - 原始数据（ground truth）
  - DASPack 压缩-解压数据
  - Adaptive2D（带噪声阈值化）压缩-解压数据

参数与原 alboran 脚本保持一致：
  sta_window=50, lta_window=500, threshold=3.0, debounce=500, time_tol=50
容差：rel_tol = 0.005（论文中使用的最粗容差，差异最显著）。

用法:
  /home/xdzhang/anaconda3/bin/python /home/xdzhang/DAS_Compression/GJI/code/sta_lta_midas.py
"""

import os
import sys
import time
import numpy as np
import h5py
import pandas as pd

# ---------------------------------------------------------------------
# 路径设置：先加入 daspack_src（本地 daspack 包）和 code 目录，
# 再导入 sta_lta_detection 中的复用函数。
# ---------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))                      # GJI/code/
_DASPACK_SRC = os.path.join(_HERE, "daspack_src")                       # GJI/code/daspack_src/
_DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "data"))        # GJI/data/
_RESULTS_DIR = os.path.normpath(os.path.join(_HERE, "..", "results"))   # GJI/results/

# 把 daspack_src 放在 path 最前，保证用本地版本而非已安装版本
sys.path.insert(0, _DASPACK_SRC)
sys.path.insert(0, _HERE)

# 导入现有函数（这些导入会顺带设置 daspack / adaptive_daspack_2d 的路径）
from sta_lta_detection import (  # noqa: E402
    detect_events,
    daspack_compress_decompress,
    adaptive_compress_decompress,
    _evaluate,
)


# =====================================================================
# 数据集列表
# =====================================================================
MIDAS_DATASETS = [
    "event_0322_a",
    "event_0322_b",
    "event_0620_a",
    "event_0620_b",
    "event_0620_c",
]


def load_midas(name):
    """加载单个 MiDAS 数据集，返回 float32 二维数组 (n_channels, n_samples)。"""
    path = os.path.join(_DATA_DIR, f"{name}.h5")
    if not os.path.exists(path):
        raise FileNotFoundError(f"MiDAS 数据文件不存在: {path}")
    with h5py.File(path, "r") as f:
        data = np.array(f["data"][:]).astype(np.float32)
    return data


def main():
    # STA/LTA 参数（与原 alboran 脚本一致）
    sta_window = 50
    lta_window = 500
    threshold = 3.0
    debounce = 500
    time_tol = 50  # 触发时间匹配容差（样本数）
    threads = 4

    # 单一相对容差：rel_tol = 0.005（最粗容差，差异最显著）
    rel_tol = 0.005

    print("=" * 90)
    print("STA/LTA 事件检测验证：MiDAS 数据集")
    print(f"  参数: STA={sta_window}, LTA={lta_window}, threshold={threshold}, "
          f"debounce={debounce}, time_tol={time_tol}")
    print(f"  rel_tol={rel_tol} (单一最粗容差)")
    print(f"  数据目录: {_DATA_DIR}")
    print(f"  结果目录: {_RESULTS_DIR}")
    print("=" * 90)

    rows = []

    for ds_name in MIDAS_DATASETS:
        print(f"\n>>> 数据集: {ds_name}")
        data = load_midas(ds_name)
        n_channels, n_samples = data.shape
        data_std = float(np.std(data))
        epsilon = rel_tol * data_std
        print(f"    shape=({n_channels}, {n_samples})  std={data_std:.4f}  "
              f"ε={epsilon:.6f}  采样率=1000 Hz  时长={n_samples/1000.0:.2f}s")

        # ---- 1) 原始数据 GT 检测 ----
        print("    [原始数据 GT] 检测...", end=" ", flush=True)
        t0 = time.perf_counter()
        gt_triggers, _ = detect_events(data, sta_window, lta_window,
                                       threshold, debounce)
        n_gt = len(gt_triggers)
        gt_detect_time = time.perf_counter() - t0
        print(f"{n_gt} events  ({gt_detect_time:.2f}s)")

        # 原始数据行（GT 自比较：TP=n_gt, FP=0, FN=0）
        rows.append({
            "dataset": ds_name,
            "n_channels": n_channels,
            "n_samples": n_samples,
            "data_std": data_std,
            "rel_tol": rel_tol,
            "epsilon": epsilon,
            "method": "original",
            "gt_events": n_gt,
            "n_events": n_gt,
            "tp": n_gt, "fp": 0, "fn": 0,
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
            "mean_abs_time_diff": 0.0,
            "std_abs_time_diff": 0.0,
            "n_matched": n_gt,
            "compress_time_s": 0.0,
        })

        # ---- 2) DASPack 压缩-解压 + 检测 ----
        print("    [DASPack] 压缩-解压...", end=" ", flush=True)
        t0 = time.perf_counter()
        data_daspack = daspack_compress_decompress(data, epsilon, threads)
        dp_compress_time = time.perf_counter() - t0
        print(f"done ({dp_compress_time:.2f}s)")
        dp_metrics = _evaluate(data_daspack, gt_triggers, sta_window,
                               lta_window, threshold, debounce, time_tol,
                               label="DASPack")
        rows.append({
            "dataset": ds_name,
            "n_channels": n_channels,
            "n_samples": n_samples,
            "data_std": data_std,
            "rel_tol": rel_tol,
            "epsilon": epsilon,
            "method": "daspack",
            "gt_events": n_gt,
            "compress_time_s": dp_compress_time,
            **dp_metrics,
        })

        # ---- 3) Adaptive2D（thresholding）压缩-解压 + 检测 ----
        print("    [Adaptive2D] 压缩-解压...", end=" ", flush=True)
        t0 = time.perf_counter()
        data_adaptive = adaptive_compress_decompress(data, epsilon, threads)
        ad_compress_time = time.perf_counter() - t0
        print(f"done ({ad_compress_time:.2f}s)")
        ad_metrics = _evaluate(data_adaptive, gt_triggers, sta_window,
                               lta_window, threshold, debounce, time_tol,
                               label="Adaptive2D(thr)")
        rows.append({
            "dataset": ds_name,
            "n_channels": n_channels,
            "n_samples": n_samples,
            "data_std": data_std,
            "rel_tol": rel_tol,
            "epsilon": epsilon,
            "method": "adaptive_2d_thresh",
            "gt_events": n_gt,
            "compress_time_s": ad_compress_time,
            **ad_metrics,
        })

    # =================================================================
    # 保存结果
    # =================================================================
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    output_csv = os.path.join(_RESULTS_DIR, "sta_lta_midas_results.csv")
    df.to_csv(output_csv, index=False)
    print(f"\n结果已保存到 {output_csv}")

    # =================================================================
    # 打印汇总表
    # =================================================================
    print("\n" + "=" * 110)
    print("MiDAS 数据集 STA/LTA 检测性能汇总 (rel_tol=0.005)")
    print("=" * 110)
    header = (f"{'dataset':>14} {'method':>20} {'GT':>6} {'n_evt':>6} "
              f"{'TP':>5} {'FP':>5} {'FN':>5} "
              f"{'Prec':>7} {'Rec':>7} {'F1':>7} "
              f"{'mean|Δt|':>9} {'t_comp':>7}")
    print(header)
    print("-" * 110)
    for _, r in df.iterrows():
        print(f"{r['dataset']:>14} {r['method']:>20} "
              f"{int(r['gt_events']):>6d} {int(r['n_events']):>6d} "
              f"{int(r['tp']):>5d} {int(r['fp']):>5d} {int(r['fn']):>5d} "
              f"{r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f} "
              f"{r['mean_abs_time_diff']:>9.4f} "
              f"{r['compress_time_s']:>7.2f}")

    # 横向对比表（DASPack vs Adaptive2D）
    print("\n" + "=" * 110)
    print("横向对比：DASPack vs Adaptive2D (rel_tol=0.005)")
    print("=" * 110)
    print(f"{'dataset':>14} "
          f"{'DP F1':>8} {'DP Rec':>8} {'DP |Δt|':>9} "
          f"{'AD F1':>8} {'AD Rec':>8} {'AD |Δt|':>9} "
          f"{'Δ F1':>7}")
    print("-" * 110)
    for ds_name in MIDAS_DATASETS:
        dp = df[(df["dataset"] == ds_name) & (df["method"] == "daspack")].iloc[0]
        ad = df[(df["dataset"] == ds_name) &
                (df["method"] == "adaptive_2d_thresh")].iloc[0]
        delta_f1 = ad["f1"] - dp["f1"]
        print(f"{ds_name:>14} "
              f"{dp['f1']:>8.4f} {dp['recall']:>8.4f} {dp['mean_abs_time_diff']:>9.4f} "
              f"{ad['f1']:>8.4f} {ad['recall']:>8.4f} {ad['mean_abs_time_diff']:>9.4f} "
              f"{delta_f1:>+7.4f}")

    # 平均值
    dp_avg = df[df["method"] == "daspack"][["f1", "recall", "mean_abs_time_diff"]].mean()
    ad_avg = df[df["method"] == "adaptive_2d_thresh"][["f1", "recall", "mean_abs_time_diff"]].mean()
    print("-" * 110)
    print(f"{'MEAN':>14} "
          f"{dp_avg['f1']:>8.4f} {dp_avg['recall']:>8.4f} {dp_avg['mean_abs_time_diff']:>9.4f} "
          f"{ad_avg['f1']:>8.4f} {ad_avg['recall']:>8.4f} {ad_avg['mean_abs_time_diff']:>9.4f} "
          f"{(ad_avg['f1'] - dp_avg['f1']):>+7.4f}")


if __name__ == "__main__":
    main()
