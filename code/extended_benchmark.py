"""
7 数据集扩展基准测试 + 统计显著性检验。

使用相对误差容限 rel_tol（ε = rel_tol × std），确保不同尺度的数据集
在相同相对误差水平下公平对比。

用法:
  cd /home/xdzhang/DAS_Compression/DASPack/improve/daspack-main/benchmark
  python extended_benchmark.py
"""

import os
import sys
import time
import numpy as np
import h5py
import struct
import pickle
import pandas as pd
from scipy import stats
from tempfile import NamedTemporaryFile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from daspack import DASCoder, Quantizer
from adaptive_daspack_2d import AdaptiveDASPack2D
from fair_benchmark import (
    DASPackFair, ZfpFair, GzipFair, JPEG2000Fair,
    FairOptions, compute_metrics, load_hdas_segment,
)


# ======================================================================
# 数据加载
# ======================================================================

DATA_DIR = "./data"

def load_all_datasets():
    """加载全部 7 个数据集。"""
    datasets = {}

    # 原始 2 个
    datasets["alboran"] = load_hdas_segment(os.path.join(DATA_DIR, "7M_S_20231005-214004_RAW.h5"))[0]
    datasets["castor"] = load_hdas_segment(os.path.join(DATA_DIR, "ZI_C_20230609-102630_RAW.h5"))[0]

    # 新增 5 个
    for name in ["event_0322_a", "event_0322_b", "event_0620_a", "event_0620_b", "event_0620_c"]:
        path = os.path.join(DATA_DIR, f"{name}.h5")
        if os.path.exists(path):
            with h5py.File(path, "r") as f:
                datasets[name] = np.array(f["data"][:]).astype(np.float32)

    return datasets


# ======================================================================
# 扩展基准测试
# ======================================================================

def run_extended_benchmark(datasets, rel_tols, block_channels=250,
                           block_samples=500, threshold_factor=1.5,
                           threads=4, output_csv="extended_benchmark_results.csv"):
    """
    在多个数据集和相对误差容限下运行基准测试。

    Parameters
    ----------
    rel_tols : list of float
        相对误差容限列表。ε = rel_tol × std(data)。
    """
    # 只对比 DASPack 和 Adaptive2D（核心对比）
    codecs_factory = {
        "daspack": lambda: DASPackFair(threads=threads),
        "adaptive_2d": lambda: AdaptiveDASPack2D(
            tolerance=0.1,
            block_channels=block_channels,
            block_samples=block_samples,
            threshold_mode="percentile",
            high_energy_fraction=0.2,
            threads=threads,
        ),
    }

    rows = []

    for ds_name, arr in datasets.items():
        data_std = float(np.std(arr))
        print(f"\n{'='*70}")
        print(f"Dataset: {ds_name}  shape={arr.shape}  std={data_std:.4f}")
        print(f"{'='*70}")

        # 预计算块分类图
        adaptive_temp = AdaptiveDASPack2D(
            tolerance=0.1,
            block_channels=min(block_channels, arr.shape[0]),
            block_samples=block_samples,
            threshold_mode="percentile",
            high_energy_fraction=0.2,
        )
        block_map, block_info = adaptive_temp.get_block_map(arr)
        n_high = sum(1 for b in block_info if b["is_high"])
        n_low = sum(1 for b in block_info if not b["is_high"])
        print(f"  Blocks: {len(block_info)} (high={n_high}, low={n_low})")

        for rel_tol in rel_tols:
            epsilon = rel_tol * data_std
            print(f"\n  rel_tol={rel_tol}  ε={epsilon:.6f}")

            opts = FairOptions(epsilon=epsilon)
            original_bytes = int(arr.nbytes)

            for codec_name, codec_factory in codecs_factory.items():
                codec = codec_factory()
                print(f"    {codec_name} ...", end=" ", flush=True)

                tmp = NamedTemporaryFile(suffix=".h5", delete=False)
                tmp_name = tmp.name
                tmp.close()

                try:
                    t0 = time.perf_counter()
                    codec.write(arr, tmp_name, opts)
                    enc_time = time.perf_counter() - t0

                    compressed_bytes = os.path.getsize(tmp_name)

                    t0 = time.perf_counter()
                    restored = codec.read(tmp_name)
                    dec_time = time.perf_counter() - t0

                    metrics = compute_metrics(arr, restored, block_map)
                    verified = metrics["max_abs_err"] <= epsilon + 1e-6
                    cr = original_bytes / compressed_bytes if compressed_bytes > 0 else float("inf")

                    row = {
                        "dataset": ds_name,
                        "codec": codec_name,
                        "rel_tol": rel_tol,
                        "epsilon": epsilon,
                        "data_std": data_std,
                        "compression_ratio": cr,
                        "original_bytes": original_bytes,
                        "compressed_bytes": compressed_bytes,
                        "enc_time_s": enc_time,
                        "dec_time_s": dec_time,
                        "verified": verified,
                        "n_blocks": len(block_info),
                        "n_high_blocks": n_high,
                        "n_low_blocks": n_low,
                        **metrics,
                    }
                    rows.append(row)

                    status = "✓" if verified else "✗"
                    print(
                        f"CR={cr:.2f}  MaxAE={metrics['max_abs_err']:.6f}  "
                        f"MSE={metrics['mse']:.2e}  "
                        f"MSE_high={metrics.get('mse_high', 0):.2e}  "
                        f"enc={enc_time:.2f}s  [{status}]"
                    )

                except Exception as e:
                    print(f"ERROR: {e}")
                    rows.append({
                        "dataset": ds_name, "codec": codec_name,
                        "rel_tol": rel_tol, "epsilon": epsilon,
                        "error": str(e),
                    })
                finally:
                    try:
                        os.remove(tmp_name)
                    except OSError:
                        pass

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\n结果已保存到 {output_csv}")
    return df


# ======================================================================
# 统计显著性检验
# ======================================================================

def statistical_analysis(df, output_csv="statistical_analysis.csv"):
    """对 MSE 改善进行统计显著性检验。"""
    print("\n" + "=" * 70)
    print("统计显著性分析")
    print("=" * 70)

    results = []

    for rel_tol in sorted(df["rel_tol"].unique()):
        subset = df[df["rel_tol"] == rel_tol]

        dp_rows = subset[subset["codec"] == "daspack"]
        ad_rows = subset[subset["codec"] == "adaptive_2d"]

        if len(dp_rows) == 0 or len(ad_rows) == 0:
            continue

        # 配对比较（按数据集配对）
        merged = pd.merge(
            dp_rows[["dataset", "compression_ratio", "mse", "mse_high", "mse_low", "snr_db", "snr_high_db"]],
            ad_rows[["dataset", "compression_ratio", "mse", "mse_high", "mse_low", "snr_db", "snr_high_db"]],
            on="dataset", suffixes=("_dp", "_ad")
        )

        n = len(merged)
        if n < 2:
            continue

        # 计算改善率
        cr_loss = (merged["compression_ratio_dp"] - merged["compression_ratio_ad"]) / merged["compression_ratio_dp"] * 100
        mse_improvement = (merged["mse_dp"] - merged["mse_ad"]) / merged["mse_dp"] * 100
        mse_high_improvement = (merged["mse_high_dp"] - merged["mse_high_ad"]) / merged["mse_high_dp"] * 100
        snr_improvement = merged["snr_db_ad"] - merged["snr_db_dp"]
        snr_high_improvement = merged["snr_high_db_ad"] - merged["snr_high_db_dp"]

        # Wilcoxon 符号秩检验（非参数配对检验）
        try:
            _, p_mse = stats.wilcoxon(merged["mse_dp"], merged["mse_ad"])
            _, p_mse_high = stats.wilcoxon(merged["mse_high_dp"], merged["mse_high_ad"])
            _, p_cr = stats.wilcoxon(merged["compression_ratio_dp"], merged["compression_ratio_ad"])
        except Exception:
            p_mse = p_mse_high = p_cr = float("nan")

        result = {
            "rel_tol": rel_tol,
            "n_datasets": n,
            "cr_loss_mean_pct": float(cr_loss.mean()),
            "cr_loss_std_pct": float(cr_loss.std()),
            "mse_improvement_mean_pct": float(mse_improvement.mean()),
            "mse_improvement_std_pct": float(mse_improvement.std()),
            "mse_high_improvement_mean_pct": float(mse_high_improvement.mean()),
            "mse_high_improvement_std_pct": float(mse_high_improvement.std()),
            "snr_improvement_mean_db": float(snr_improvement.mean()),
            "snr_high_improvement_mean_db": float(snr_high_improvement.mean()),
            "p_value_mse": float(p_mse),
            "p_value_mse_high": float(p_mse_high),
            "p_value_cr": float(p_cr),
            "significant_mse": p_mse < 0.05 if not np.isnan(p_mse) else False,
            "significant_mse_high": p_mse_high < 0.05 if not np.isnan(p_mse_high) else False,
        }
        results.append(result)

        print(f"\n  rel_tol={rel_tol} (n={n} datasets)")
        print(f"    CR loss:       {cr_loss.mean():.2f}% ± {cr_loss.std():.2f}%")
        print(f"    MSE improvement: {mse_improvement.mean():.2f}% ± {mse_improvement.std():.2f}%  (p={p_mse:.4f})")
        print(f"    MSE_high improvement: {mse_high_improvement.mean():.2f}% ± {mse_high_improvement.std():.2f}%  (p={p_mse_high:.4f})")
        print(f"    SNR improvement: {snr_improvement.mean():.2f} dB")
        print(f"    SNR_high improvement: {snr_high_improvement.mean():.2f} dB")
        sig_mse = "***" if p_mse < 0.001 else ("**" if p_mse < 0.01 else ("*" if p_mse < 0.05 else "ns"))
        sig_mse_h = "***" if p_mse_high < 0.001 else ("**" if p_mse_high < 0.01 else ("*" if p_mse_high < 0.05 else "ns"))
        print(f"    Significance: MSE={sig_mse}, MSE_high={sig_mse_h}")

    stat_df = pd.DataFrame(results)
    stat_df.to_csv(output_csv, index=False)
    print(f"\n统计结果已保存到 {output_csv}")
    return stat_df


# ======================================================================
# 主入口
# ======================================================================

if __name__ == "__main__":
    print("加载所有数据集...")
    datasets = load_all_datasets()
    print(f"共 {len(datasets)} 个数据集:")
    for name, arr in datasets.items():
        print(f"  {name}: {arr.shape}  std={np.std(arr):.4f}")

    # 相对误差容限（ε = rel_tol × std）
    # 对于 std=100 的数据，对应 ε = 0.005, 0.01, 0.02, 0.05, 0.1, 0.2
    rel_tols = [0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002]

    # 运行基准测试
    df = run_extended_benchmark(
        datasets=datasets,
        rel_tols=rel_tols,
        block_channels=250,
        block_samples=500,
        threshold_factor=1.5,
        threads=4,
        output_csv="extended_benchmark_results.csv",
    )

    # 统计显著性检验
    stat_df = statistical_analysis(df)

    # 汇总表
    print("\n" + "=" * 80)
    print("汇总：各数据集压缩率对比（rel_tol=0.001, 即 ε=0.1×std）")
    print("=" * 80)
    subset = df[df["rel_tol"] == 0.001]
    if len(subset) > 0:
        pivot = subset.pivot_table(
            index="dataset", columns="codec",
            values="compression_ratio", aggfunc="first"
        )
        print(pivot.to_string(float_format="%.2f"))

    print("\n" + "=" * 80)
    print("汇总：各数据集高能量区域 MSE 对比（rel_tol=0.001）")
    print("=" * 80)
    if len(subset) > 0 and "mse_high" in subset.columns:
        pivot = subset.pivot_table(
            index="dataset", columns="codec",
            values="mse_high", aggfunc="first"
        )
        print(pivot.to_string(float_format="%.2e"))

    print("\n" + "=" * 80)
    print("总体结论（所有 rel_tol 平均）")
    print("=" * 80)
    for metric, label in [("mse_improvement_mean_pct", "MSE 改善"),
                          ("mse_high_improvement_mean_pct", "高能量 MSE 改善"),
                          ("cr_loss_mean_pct", "CR 损失")]:
        vals = stat_df[metric].dropna()
        if len(vals) > 0:
            print(f"  {label}: {vals.mean():.2f}% (range: {vals.min():.2f}% - {vals.max():.2f}%)")

    p_vals = stat_df["p_value_mse_high"].dropna()
    if len(p_vals) > 0:
        print(f"  高能量 MSE p-value 范围: {p_vals.min():.6f} - {p_vals.max():.6f}")
        print(f"  所有 rel_tol 下高能量 MSE 改善均显著: {all(p < 0.05 for p in p_vals)}")
