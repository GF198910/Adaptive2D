"""
STA/LTA 地震事件检测对比脚本。

对 alboran 数据集运行经典 STA/LTA 检测，对比原始数据、DASPack 压缩-解压数据
和 Adaptive2D（带噪声阈值化）压缩-解压数据上的检测结果。

算法参数：
  - STA 窗口 = 50 samples
  - LTA 窗口 = 500 samples
  - 触发阈值 = 3.0
  - 对每个通道独立运行检测，记录触发时间（样本索引）

用法:
  cd /home/xdzhang/DAS_Compression/DASPack/improve/daspack-main/benchmark
  /home/xdzhang/anaconda3/bin/python sta_lta_detection.py
"""

import os
import sys
import time
import numpy as np
import h5py
import pandas as pd

# 添加路径：父目录用于导入 daspack 包，当前目录用于导入 adaptive_daspack_2d
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # daspack-main/
sys.path.insert(0, _HERE)                   # benchmark/

from daspack import DASCoder, Quantizer
from adaptive_daspack_2d import AdaptiveDASPack2D


# ======================================================================
# 数据加载
# ======================================================================

DATA_DIR = os.path.join(_HERE, "data")


def load_alboran():
    """加载 alboran 数据集。"""
    path = os.path.join(DATA_DIR, "7M_S_20231005-214004_RAW.h5")
    if not os.path.exists(path):
        raise FileNotFoundError(f"alboran 数据文件不存在: {path}")
    with h5py.File(path, "r") as f:
        data = np.array(f["data"][:]).astype(np.float32)
    return data


# ======================================================================
# 经典 STA/LTA 检测
# ======================================================================

def sta_lta(signal, sta_window=50, lta_window=500, threshold=3.0,
            debounce_samples=500):
    """
    经典 STA/LTA 事件检测（向量化实现）。

    特征函数 cf = signal^2。
    STA[i] = mean(cf[i-sta:i]), LTA[i] = mean(cf[i-lta:i])。
    当 STA/LTA > threshold 且满足去抖动间隔时记录一次触发。

    Parameters
    ----------
    signal : 1D ndarray
        单通道时间序列。
    sta_window : int
        短时窗口长度（样本数）。
    lta_window : int
        长时窗口长度（样本数）。
    threshold : float
        触发阈值。
    debounce_samples : int
        两次触发之间的最小间隔（去抖动），避免同一事件多次触发。

    Returns
    -------
    triggers : list of int
        触发时刻的样本索引。
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    if n < lta_window + sta_window:
        return []

    cf = signal ** 2
    csum = np.concatenate(([0.0], np.cumsum(cf)))  # length n+1, csum[k]=sum(cf[:k])

    # 对 i ∈ [lta_window, n-1] 计算 STA/LTA
    i_arr = np.arange(lta_window, n)
    sta = (csum[i_arr] - csum[i_arr - sta_window]) / sta_window
    lta = (csum[i_arr] - csum[i_arr - lta_window]) / lta_window

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(lta > 0, sta / lta, 0.0)

    above_idx = np.where(ratio > threshold)[0]
    trigger_indices = above_idx + lta_window

    # 去抖动：保留第一个触发，跳过后续 debounce_samples 内的触发
    triggers = []
    last = -debounce_samples - 1
    for t in trigger_indices:
        if t - last >= debounce_samples:
            triggers.append(int(t))
            last = t
    return triggers


def detect_events(data, sta_window=50, lta_window=500, threshold=3.0,
                  debounce_samples=500):
    """
    对每个通道运行 STA/LTA 检测。

    Returns
    -------
    triggers : list of (channel, sample)
        所有通道的触发事件列表。
    per_channel : list of list of int
        每个通道的触发时间列表。
    """
    n_channels, n_samples = data.shape
    per_channel = []
    triggers = []
    for ch in range(n_channels):
        ch_trigs = sta_lta(data[ch], sta_window, lta_window,
                           threshold, debounce_samples)
        per_channel.append(ch_trigs)
        for t in ch_trigs:
            triggers.append((ch, int(t)))
    return triggers, per_channel


# ======================================================================
# 触发事件匹配与混淆矩阵
# ======================================================================

def match_triggers(gt_triggers, comp_triggers, time_tol=50):
    """
    将压缩数据的触发事件与原始数据（ground truth）匹配。

    匹配规则：同一通道内，压缩触发与最近的未匹配 GT 触发的时间差 ≤ time_tol
    则记为一次匹配 (TP)。

    Returns
    -------
    tp : int  真阳性（匹配成功）
    fp : int  假阳性（压缩多检）
    fn : int  假阴性（压缩漏检）
    matched_pairs : list of (gt_time, comp_time, diff)
        匹配对的详细信息，diff = comp_time - gt_time。
    """
    gt_by_ch = {}
    for ch, t in gt_triggers:
        gt_by_ch.setdefault(ch, []).append(t)
    comp_by_ch = {}
    for ch, t in comp_triggers:
        comp_by_ch.setdefault(ch, []).append(t)

    tp = 0
    fp = 0
    fn = 0
    matched_pairs = []

    all_channels = set(gt_by_ch.keys()) | set(comp_by_ch.keys())
    for ch in all_channels:
        gt_times = sorted(gt_by_ch.get(ch, []))
        comp_times = sorted(comp_by_ch.get(ch, []))
        gt_used = [False] * len(gt_times)

        for ct in comp_times:
            best = -1
            best_diff = time_tol + 1
            for j, gtt in enumerate(gt_times):
                if gt_used[j]:
                    continue
                d = abs(ct - gtt)
                if d <= time_tol and d < best_diff:
                    best = j
                    best_diff = d
            if best >= 0:
                gt_used[best] = True
                tp += 1
                matched_pairs.append((gt_times[best], ct, ct - gt_times[best]))
            else:
                fp += 1
        fn += sum(1 for u in gt_used if not u)

    return tp, fp, fn, matched_pairs


def confusion_metrics(tp, fp, fn):
    """计算检测率、精确率、F1。"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


# ======================================================================
# 压缩-解压
# ======================================================================

def daspack_compress_decompress(arr, epsilon, threads=4):
    """DASPack：step = 2ε。"""
    coder = DASCoder(threads=threads)
    step = 2.0 * epsilon
    buf = coder.encode(arr.astype(np.float64), Quantizer.Uniform(step=step))
    restored = coder.decode(buf)
    return restored


def adaptive_compress_decompress(arr, epsilon, threads=4):
    """Adaptive2D with thresholding。"""
    codec = AdaptiveDASPack2D(
        tolerance=epsilon,
        block_channels=250,
        block_samples=500,
        threshold_mode="percentile",
        high_energy_fraction=0.2,
        noise_thresholding=True,
        threads=threads,
    )
    buf = codec.compress(arr)
    restored = codec.decompress(buf)
    return restored


# ======================================================================
# 主流程
# ======================================================================

def _time_diff_stats(pairs):
    """触发时间差异统计：返回 (mean_abs, std_abs, n_matched)。"""
    if not pairs:
        return 0.0, 0.0, 0
    diffs = np.array([p[2] for p in pairs], dtype=np.float64)
    return (float(np.mean(np.abs(diffs))),
            float(np.std(np.abs(diffs))),
            len(diffs))


def _evaluate(restored, gt_triggers, sta_window, lta_window, threshold,
              debounce, time_tol, label=""):
    """对一份重建数据运行 STA/LTA 检测并与 GT 匹配，返回指标 dict。"""
    t0 = time.perf_counter()
    triggers, _ = detect_events(restored, sta_window, lta_window,
                                threshold, debounce)
    tp, fp, fn, pairs = match_triggers(gt_triggers, triggers, time_tol)
    prec, rec, f1 = confusion_metrics(tp, fp, fn)
    mean_abs, std_abs, n_matched = _time_diff_stats(pairs)
    if label:
        print(f"    {label}: {len(triggers)} events, "
              f"TP={tp}, FP={fp}, FN={fn}, "
              f"Recall={rec:.4f}, F1={f1:.4f} "
              f"({time.perf_counter() - t0:.2f}s)")
    return {
        "n_events": len(triggers),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": prec, "recall": rec, "f1": f1,
        "mean_abs_time_diff": mean_abs,
        "std_abs_time_diff": std_abs,
        "n_matched": n_matched,
    }


def main():
    # STA/LTA 参数（保持一致）
    sta_window = 50
    lta_window = 500
    threshold = 3.0
    debounce = 500
    time_tol = 50  # 触发时间匹配容差（样本数）
    threads = 4

    # 多个相对容限（rel_tol = ε / std）
    rel_tols = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005]

    print("加载 alboran 数据集...")
    data = load_alboran()
    data_std = float(np.std(data))
    print(f"  shape={data.shape}  std={data_std:.4f}")

    # ------------------------------------------------------------------
    # 原始数据检测（作为 ground truth，与容限无关，只算一次）
    # ------------------------------------------------------------------
    print("\n运行 STA/LTA 检测 (ground truth = 原始数据)...")
    print(f"  参数: STA={sta_window}, LTA={lta_window}, "
          f"threshold={threshold}, debounce={debounce}, time_tol={time_tol}")

    print("  [原始数据] ...", end=" ", flush=True)
    t0 = time.perf_counter()
    gt_triggers, _ = detect_events(data, sta_window, lta_window,
                                   threshold, debounce)
    n_gt = len(gt_triggers)
    print(f"{n_gt} events ({time.perf_counter() - t0:.2f}s)")

    # ------------------------------------------------------------------
    # 在多个容限下运行压缩-解压 + 检测
    # ------------------------------------------------------------------
    rows = []
    for idx, rel_tol in enumerate(rel_tols, start=1):
        epsilon = rel_tol * data_std
        print(f"\n[{idx}/{len(rel_tols)}] rel_tol={rel_tol:.4f}  "
              f"ε={epsilon:.6f}")

        # 原始数据行（与容限无关，固定为 GT）
        base = {
            "rel_tol": rel_tol,
            "epsilon": epsilon,
            "method": "original",
            "n_events": n_gt,
            "tp": n_gt, "fp": 0, "fn": 0,
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
            "mean_abs_time_diff": 0.0, "std_abs_time_diff": 0.0,
            "n_matched": n_gt,
        }
        rows.append(base)

        # DASPack：step = 2ε
        print("  压缩-解压 DASPack ...", end=" ", flush=True)
        t0 = time.perf_counter()
        data_daspack = daspack_compress_decompress(data, epsilon, threads)
        print(f"done ({time.perf_counter() - t0:.2f}s)")
        dp_metrics = _evaluate(data_daspack, gt_triggers, sta_window,
                               lta_window, threshold, debounce, time_tol,
                               label="DASPack")
        rows.append({
            "rel_tol": rel_tol, "epsilon": epsilon,
            "method": "daspack", **dp_metrics,
        })

        # Adaptive2D（with thresholding=True）
        print("  压缩-解压 Adaptive2D (thresholding) ...", end=" ", flush=True)
        t0 = time.perf_counter()
        data_adaptive = adaptive_compress_decompress(data, epsilon, threads)
        print(f"done ({time.perf_counter() - t0:.2f}s)")
        ad_metrics = _evaluate(data_adaptive, gt_triggers, sta_window,
                               lta_window, threshold, debounce, time_tol,
                               label="Adaptive2D(thr)")
        rows.append({
            "rel_tol": rel_tol, "epsilon": epsilon,
            "method": "adaptive_2d_thresh", **ad_metrics,
        })

    # ------------------------------------------------------------------
    # 保存结果
    # ------------------------------------------------------------------
    df = pd.DataFrame(rows)
    output_csv = os.path.join(_HERE, "sta_lta_multi_tolerance.csv")
    df.to_csv(output_csv, index=False)
    print(f"\n结果已保存到 {output_csv}")

    # ------------------------------------------------------------------
    # 打印汇总表
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("STA/LTA 检测性能 vs 相对容限 (alboran, ground truth = 原始数据)")
    print(f"  参数: STA={sta_window}, LTA={lta_window}, threshold={threshold}, "
          f"debounce={debounce}, time_tol={time_tol}")
    print(f"  原始数据 GT 事件数: {n_gt}")
    print("=" * 100)

    for method in ["daspack", "adaptive_2d_thresh"]:
        sub = df[df["method"] == method].copy()
        print(f"\n--- {method} ---")
        print(f"{'rel_tol':>10} {'ε':>12} {'n_events':>9} "
              f"{'TP':>6} {'FP':>7} {'FN':>6} "
              f"{'Precision':>10} {'Recall':>9} {'F1':>8} "
              f"{'mean|Δt|':>10}")
        print("-" * 100)
        for _, r in sub.iterrows():
            print(f"{r['rel_tol']:>10.4f} {r['epsilon']:>12.6f} "
                  f"{int(r['n_events']):>9d} "
                  f"{int(r['tp']):>6d} {int(r['fp']):>7d} {int(r['fn']):>6d} "
                  f"{r['precision']:>10.4f} {r['recall']:>9.4f} "
                  f"{r['f1']:>8.4f} {r['mean_abs_time_diff']:>10.4f}")

    # 横向对比 F1 随容限变化
    print("\n" + "=" * 100)
    print("F1 / Recall 横向对比")
    print("=" * 100)
    print(f"{'rel_tol':>10} {'ε':>12} "
          f"{'DASPack F1':>12} {'DASPack Rec':>12} "
          f"{'Adaptive F1':>12} {'Adaptive Rec':>13}")
    print("-" * 100)
    for rel_tol in rel_tols:
        dp = df[(df["rel_tol"] == rel_tol) & (df["method"] == "daspack")].iloc[0]
        ad = (df[(df["rel_tol"] == rel_tol) &
                 (df["method"] == "adaptive_2d_thresh")].iloc[0])
        eps = rel_tol * data_std
        print(f"{rel_tol:>10.4f} {eps:>12.6f} "
              f"{dp['f1']:>12.4f} {dp['recall']:>12.4f} "
              f"{ad['f1']:>12.4f} {ad['recall']:>13.4f}")


if __name__ == "__main__":
    main()
