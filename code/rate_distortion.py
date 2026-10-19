"""
率失真分析：绘制 CR vs MSE / CR vs SNR 曲线。

在多个 ε 值下对比 DASPack 和 Adaptive2D 的率失真性能，
证明自适应方法在相同压缩率下有更低的 MSE（尤其是在高能量区域），
或在相同 MSE 下有更高的压缩率。

用法:
  cd /home/xdzhang/DAS_Compression/DASPack/improve/daspack-main/benchmark
  python rate_distortion.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fair_benchmark import (
    load_hdas_segment,
    DASPackFair,
    AdaptiveDASPack2D,
    FairOptions,
    compute_metrics,
)
from adaptive_daspack_2d import AdaptiveDASPack2D as AD2D
import time
from tempfile import NamedTemporaryFile


def run_rate_distortion(datasets, epsilons, block_channels=500, block_samples=500,
                        threshold_factor=1.5, threads=4):
    """
    对多个 ε 值运行压缩并收集率失真数据。

    Returns
    -------
    results : list of dict
        每条记录包含 dataset, codec, epsilon, cr, mse, snr, mse_high, mse_low 等。
    """
    results = []

    for ds_name, arr in datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}  shape={arr.shape}")
        print(f"{'='*60}")

        # 预计算块分类图
        adaptive_temp = AD2D(
            tolerance=0.1,
            block_channels=block_channels,
            block_samples=block_samples,
            threshold_factor=threshold_factor,
        )
        block_map, _ = adaptive_temp.get_block_map(arr)

        # 初始化压缩器
        daspack = DASPackFair(threads=threads)
        adaptive = AdaptiveDASPack2D(
            tolerance=0.1,
            block_channels=block_channels,
            block_samples=block_samples,
            threshold_factor=threshold_factor,
            threads=threads,
        )

        original_bytes = int(arr.nbytes)

        for epsilon in epsilons:
            print(f"  ε = {epsilon} ...", end=" ", flush=True)
            opts = FairOptions(epsilon=epsilon)

            for codec, name in [(daspack, "daspack"), (adaptive, "adaptive_2d")]:
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
                    cr = original_bytes / compressed_bytes

                    results.append({
                        "dataset": ds_name,
                        "codec": name,
                        "epsilon": epsilon,
                        "cr": cr,
                        "compressed_bytes": compressed_bytes,
                        "enc_time": enc_time,
                        "dec_time": dec_time,
                        **metrics,
                    })
                finally:
                    try:
                        os.remove(tmp_name)
                    except OSError:
                        pass

            print("done")

    return pd.DataFrame(results)


def plot_rate_distortion(df, output_dir="."):
    """绘制率失真曲线。"""
    datasets = df["dataset"].unique()
    codecs = df["codec"].unique()
    colors = {"daspack": "#1f77b4", "adaptive_2d": "#ff7f0e"}
    markers = {"daspack": "s", "adaptive_2d": "o"}

    for ds in datasets:
        ds_df = df[df["dataset"] == ds]

        # --- CR vs MSE ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for codec in codecs:
            cdf = ds_df[ds_df["codec"] == codec].sort_values("epsilon")
            color = colors.get(codec, "gray")
            marker = markers.get(codec, "x")

            # Overall MSE
            axes[0].plot(cdf["cr"], cdf["mse"], marker=marker, color=color,
                         label=codec, linewidth=2, markersize=8)
            # High-energy MSE
            if "mse_high" in cdf.columns:
                axes[1].plot(cdf["cr"], cdf["mse_high"], marker=marker, color=color,
                             label=codec, linewidth=2, markersize=8)
            # Low-energy MSE
            if "mse_low" in cdf.columns:
                axes[2].plot(cdf["cr"], cdf["mse_low"], marker=marker, color=color,
                             label=codec, linewidth=2, markersize=8)

        titles = ["Overall MSE", "High-Energy Region MSE", "Low-Energy Region MSE"]
        ylabels = ["MSE", "MSE (high-energy)", "MSE (low-energy)"]
        for ax, title, ylabel in zip(axes, titles, ylabels):
            ax.set_xlabel("Compression Ratio")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ds} - {title}")
            ax.set_yscale("log")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = os.path.join(output_dir, f"rate_distortion_{ds}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  保存: {fname}")

        # --- CR vs SNR ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for codec in codecs:
            cdf = ds_df[ds_df["codec"] == codec].sort_values("epsilon")
            color = colors.get(codec, "gray")
            marker = markers.get(codec, "x")

            axes[0].plot(cdf["cr"], cdf["snr_db"], marker=marker, color=color,
                         label=codec, linewidth=2, markersize=8)
            if "snr_high_db" in cdf.columns:
                axes[1].plot(cdf["cr"], cdf["snr_high_db"], marker=marker, color=color,
                             label=codec, linewidth=2, markersize=8)
            if "snr_low_db" in cdf.columns:
                axes[2].plot(cdf["cr"], cdf["snr_low_db"], marker=marker, color=color,
                             label=codec, linewidth=2, markersize=8)

        titles = ["Overall SNR", "High-Energy Region SNR", "Low-Energy Region SNR"]
        ylabels = ["SNR (dB)", "SNR (high) (dB)", "SNR (low) (dB)"]
        for ax, title, ylabel in zip(axes, titles, ylabels):
            ax.set_xlabel("Compression Ratio")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ds} - {title}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = os.path.join(output_dir, f"rate_distortion_snr_{ds}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  保存: {fname}")


if __name__ == "__main__":
    data_dir = "./data"

    datasets = {
        "alboran": load_hdas_segment(os.path.join(data_dir, "7M_S_20231005-214004_RAW.h5"))[0],
        "castor": load_hdas_segment(os.path.join(data_dir, "ZI_C_20230609-102630_RAW.h5"))[0],
    }

    epsilons = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

    df = run_rate_distortion(
        datasets=datasets,
        epsilons=epsilons,
        block_channels=500,
        block_samples=500,
        threshold_factor=1.5,
        threads=4,
    )

    df.to_csv("rate_distortion_results.csv", index=False)
    print(f"\n数据已保存到 rate_distortion_results.csv")

    plot_rate_distortion(df, output_dir=".")

    # 打印表格
    print("\n" + "=" * 80)
    print("率失真数据汇总")
    print("=" * 80)
    for ds in df["dataset"].unique():
        print(f"\n--- {ds} ---")
        ds_df = df[df["dataset"] == ds]
        print(ds_df[["codec", "epsilon", "cr", "mse", "mse_high", "mse_low",
                      "snr_db", "snr_high_db", "snr_low_db"]].to_string(index=False))
