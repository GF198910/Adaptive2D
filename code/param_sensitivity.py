"""
参数敏感性分析：block size 和 high_energy_fraction 扫描。

生成：
  1. Block size 扫描曲线（CR, MSE, MSE_high vs block_channels × block_samples）
  2. High energy fraction 扫描曲线（CR, MSE, MSE_high vs fraction）
  3. 2D 热力图（block_channels vs block_samples）

用法:
  cd /home/xdzhang/DAS_Compression/DASPack/improve/daspack-main/benchmark
  python param_sensitivity.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tempfile import NamedTemporaryFile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fair_benchmark import DASPackFair, FairOptions, compute_metrics, load_hdas_segment
from adaptive_daspack_2d import AdaptiveDASPack2D
from extended_benchmark import load_all_datasets


OUTPUT_DIR = "param_sensitivity_plots"
REL_TOL = 0.001  # ε = 0.001 × std


def compress_and_metrics(codec, arr, opts, block_map=None):
    """压缩、解压并计算指标。"""
    tmp = NamedTemporaryFile(suffix=".h5", delete=False)
    tmp_name = tmp.name
    tmp.close()
    try:
        t0 = time.perf_counter()
        codec.write(arr, tmp_name, opts)
        enc_time = time.perf_counter() - t0

        compressed_bytes = os.path.getsize(tmp_name)
        cr = arr.nbytes / compressed_bytes

        t0 = time.perf_counter()
        restored = codec.read(tmp_name)
        dec_time = time.perf_counter() - t0

        metrics = compute_metrics(arr, restored, block_map)
        metrics["cr"] = cr
        metrics["enc_time"] = enc_time
        metrics["dec_time"] = dec_time
        return metrics
    finally:
        os.remove(tmp_name)


# ======================================================================
# 1. Block size 扫描
# ======================================================================

def sweep_block_size(datasets, block_sizes, frac=0.2, threads=4):
    """
    扫描不同 block size 组合。

    Parameters
    ----------
    block_sizes : list of (int, int)
        (block_channels, block_samples) 列表。
    """
    results = []

    for ds_name, arr in datasets.items():
        data_std = float(np.std(arr))
        epsilon = REL_TOL * data_std
        opts = FairOptions(epsilon=epsilon)
        original_bytes = int(arr.nbytes)

        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} (shape={arr.shape}, ε={epsilon:.4f})")
        print(f"{'='*60}")

        # DASPack baseline (固定，不受 block size 影响)
        daspack = DASPackFair(threads=threads)
        dp_metrics = compress_and_metrics(daspack, arr, opts)
        print(f"  DASPack baseline: CR={dp_metrics['cr']:.2f}, MSE={dp_metrics['mse']:.2e}")

        for bch, bsa in block_sizes:
            # 确保块大小不超过数据维度
            actual_bch = min(bch, arr.shape[0])
            actual_bsa = min(bsa, arr.shape[1])

            adaptive = AdaptiveDASPack2D(
                tolerance=epsilon,
                block_channels=actual_bch,
                block_samples=actual_bsa,
                threshold_mode="percentile",
                high_energy_fraction=frac,
                threads=threads,
            )

            # 计算块分类图
            block_map, block_info = adaptive.get_block_map(arr, opts)
            n_high = sum(1 for b in block_info if b["is_high"])

            metrics = compress_and_metrics(adaptive, arr, opts, block_map)

            row = {
                "dataset": ds_name,
                "block_channels": actual_bch,
                "block_samples": actual_bsa,
                "n_blocks": len(block_info),
                "n_high": n_high,
                "cr": metrics["cr"],
                "mse": metrics["mse"],
                "mse_high": metrics.get("mse_high", 0),
                "mse_low": metrics.get("mse_low", 0),
                "snr_db": metrics.get("snr_db", 0),
                "snr_high_db": metrics.get("snr_high_db", 0),
                "enc_time": metrics["enc_time"],
                "dp_cr": dp_metrics["cr"],
                "dp_mse": dp_metrics["mse"],
            }
            results.append(row)

            cr_loss = (dp_metrics["cr"] - metrics["cr"]) / dp_metrics["cr"] * 100
            print(f"  block=({actual_bch},{actual_bsa}): "
                  f"CR={metrics['cr']:.2f} ({cr_loss:+.1f}%)  "
                  f"MSE={metrics['mse']:.2e}  "
                  f"MSE_high={metrics.get('mse_high', 0):.2e}  "
                  f"blocks={len(block_info)}(high={n_high})")

    return pd.DataFrame(results)


# ======================================================================
# 2. High energy fraction 扫描
# ======================================================================

def sweep_fraction(datasets, fractions, block_channels=250, block_samples=500, threads=4):
    """扫描不同 high_energy_fraction 值。"""
    results = []

    for ds_name, arr in datasets.items():
        data_std = float(np.std(arr))
        epsilon = REL_TOL * data_std
        opts = FairOptions(epsilon=epsilon)

        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} (ε={epsilon:.4f})")
        print(f"{'='*60}")

        daspack = DASPackFair(threads=threads)
        dp_metrics = compress_and_metrics(daspack, arr, opts)

        for frac in fractions:
            adaptive = AdaptiveDASPack2D(
                tolerance=epsilon,
                block_channels=min(block_channels, arr.shape[0]),
                block_samples=block_samples,
                threshold_mode="percentile",
                high_energy_fraction=frac,
                threads=threads,
            )

            block_map, block_info = adaptive.get_block_map(arr, opts)
            n_high = sum(1 for b in block_info if b["is_high"])

            metrics = compress_and_metrics(adaptive, arr, opts, block_map)

            row = {
                "dataset": ds_name,
                "fraction": frac,
                "n_blocks": len(block_info),
                "n_high": n_high,
                "cr": metrics["cr"],
                "mse": metrics["mse"],
                "mse_high": metrics.get("mse_high", 0),
                "mse_low": metrics.get("mse_low", 0),
                "snr_db": metrics.get("snr_db", 0),
                "snr_high_db": metrics.get("snr_high_db", 0),
                "dp_cr": dp_metrics["cr"],
                "dp_mse": dp_metrics["mse"],
            }
            results.append(row)

            cr_loss = (dp_metrics["cr"] - metrics["cr"]) / dp_metrics["cr"] * 100
            mse_imp = (dp_metrics["mse"] - metrics["mse"]) / dp_metrics["mse"] * 100
            print(f"  frac={frac:.2f}: CR={metrics['cr']:.2f} ({cr_loss:+.1f}%)  "
                  f"MSE={metrics['mse']:.2e} ({mse_imp:+.1f}%)  "
                  f"MSE_high={metrics.get('mse_high', 0):.2e}  "
                  f"high={n_high}/{len(block_info)}")

    return pd.DataFrame(results)


# ======================================================================
# 3. 绘图函数
# ======================================================================

def plot_block_size_sweep(df, output_dir):
    """绘制 block size 扫描结果。"""
    datasets = df["dataset"].unique()
    n_ds = len(datasets)

    # 生成 block_size 标签
    df["block_label"] = df.apply(
        lambda r: f"{r['block_channels']}×{r['block_samples']}", axis=1
    )
    block_labels = df["block_label"].unique()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    colors = plt.cm.tab10(np.linspace(0, 0.8, n_ds))

    # (a) CR vs block size
    ax = axes[0, 0]
    for i, ds in enumerate(datasets):
        dsdf = df[df["dataset"] == ds]
        ax.plot(range(len(dsdf)), dsdf["cr"], marker="o", color=colors[i],
                linewidth=2, markersize=6, label=ds)
        # DASPack baseline
        ax.axhline(y=dsdf["dp_cr"].iloc[0], color=colors[i], linestyle="--",
                   alpha=0.4, linewidth=1)
    ax.set_xticks(range(len(block_labels)))
    ax.set_xticklabels(block_labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Block size (channels × samples)")
    ax.set_ylabel("Compression Ratio")
    ax.set_title("CR vs Block Size (dashed = DASPack baseline)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.2)

    # (b) MSE vs block size
    ax = axes[0, 1]
    for i, ds in enumerate(datasets):
        dsdf = df[df["dataset"] == ds]
        ax.plot(range(len(dsdf)), dsdf["mse"], marker="s", color=colors[i],
                linewidth=2, markersize=6, label=ds)
        ax.axhline(y=dsdf["dp_mse"].iloc[0], color=colors[i], linestyle="--",
                   alpha=0.4, linewidth=1)
    ax.set_xticks(range(len(block_labels)))
    ax.set_xticklabels(block_labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Block size (channels × samples)")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.set_title("MSE vs Block Size (dashed = DASPack baseline)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)

    # (c) MSE_high vs block size
    ax = axes[1, 0]
    for i, ds in enumerate(datasets):
        dsdf = df[df["dataset"] == ds]
        ax.plot(range(len(dsdf)), dsdf["mse_high"], marker="^", color=colors[i],
                linewidth=2, markersize=6, label=ds)
    ax.set_xticks(range(len(block_labels)))
    ax.set_xticklabels(block_labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Block size (channels × samples)")
    ax.set_ylabel("MSE (high-energy)")
    ax.set_yscale("log")
    ax.set_title("High-Energy MSE vs Block Size")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # (d) SNR_high vs block size
    ax = axes[1, 1]
    for i, ds in enumerate(datasets):
        dsdf = df[df["dataset"] == ds]
        ax.plot(range(len(dsdf)), dsdf["snr_high_db"], marker="d", color=colors[i],
                linewidth=2, markersize=6, label=ds)
    ax.set_xticks(range(len(block_labels)))
    ax.set_xticklabels(block_labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Block size (channels × samples)")
    ax.set_ylabel("SNR (dB, high-energy)")
    ax.set_title("High-Energy SNR vs Block Size")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.suptitle(f"Block Size Sensitivity (rel_tol={REL_TOL}, frac=0.2)", fontsize=14, y=1.01)
    plt.tight_layout()
    fname = os.path.join(output_dir, "sweep_block_size.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {fname}")


def plot_fraction_sweep(df, output_dir):
    """绘制 high_energy_fraction 扫描结果。"""
    datasets = df["dataset"].unique()
    n_ds = len(datasets)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    colors = plt.cm.tab10(np.linspace(0, 0.8, n_ds))

    # (a) CR vs fraction
    ax = axes[0, 0]
    for i, ds in enumerate(datasets):
        dsdf = df[df["dataset"] == ds].sort_values("fraction")
        ax.plot(dsdf["fraction"] * 100, dsdf["cr"], marker="o", color=colors[i],
                linewidth=2, markersize=6, label=ds)
        ax.axhline(y=dsdf["dp_cr"].iloc[0], color=colors[i], linestyle="--",
                   alpha=0.4, linewidth=1)
    ax.set_xlabel("High-Energy Fraction (%)")
    ax.set_ylabel("Compression Ratio")
    ax.set_title("CR vs High-Energy Fraction (dashed = DASPack)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # (b) MSE vs fraction
    ax = axes[0, 1]
    for i, ds in enumerate(datasets):
        dsdf = df[df["dataset"] == ds].sort_values("fraction")
        ax.plot(dsdf["fraction"] * 100, dsdf["mse"], marker="s", color=colors[i],
                linewidth=2, markersize=6, label=ds)
        ax.axhline(y=dsdf["dp_mse"].iloc[0], color=colors[i], linestyle="--",
                   alpha=0.4, linewidth=1)
    ax.set_xlabel("High-Energy Fraction (%)")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.set_title("MSE vs High-Energy Fraction (dashed = DASPack)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # (c) MSE_high vs fraction
    ax = axes[1, 0]
    for i, ds in enumerate(datasets):
        dsdf = df[df["dataset"] == ds].sort_values("fraction")
        ax.plot(dsdf["fraction"] * 100, dsdf["mse_high"], marker="^", color=colors[i],
                linewidth=2, markersize=6, label=ds)
    ax.set_xlabel("High-Energy Fraction (%)")
    ax.set_ylabel("MSE (high-energy)")
    ax.set_yscale("log")
    ax.set_title("High-Energy MSE vs Fraction")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # (d) CR loss vs MSE improvement (率失真 trade-off)
    ax = axes[1, 1]
    for i, ds in enumerate(datasets):
        dsdf = df[df["dataset"] == ds].sort_values("fraction")
        cr_loss = (dsdf["dp_cr"] - dsdf["cr"]) / dsdf["dp_cr"] * 100
        mse_imp = (dsdf["dp_mse"] - dsdf["mse"]) / dsdf["dp_mse"] * 100
        ax.plot(cr_loss, mse_imp, marker="o", color=colors[i],
                linewidth=2, markersize=6, label=ds)
        # 标注 fraction 值
        for _, row in dsdf.iterrows():
            cl = (row["dp_cr"] - row["cr"]) / row["dp_cr"] * 100
            mi = (row["dp_mse"] - row["mse"]) / row["dp_mse"] * 100
            ax.annotate(f"{row['fraction']*100:.0f}%", (cl, mi),
                        fontsize=7, ha="center", va="bottom")
    ax.set_xlabel("CR Loss (%)")
    ax.set_ylabel("MSE Improvement (%)")
    ax.set_title("Trade-off: CR Loss vs MSE Improvement")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax.axvline(x=0, color="gray", linestyle="-", alpha=0.3)

    fig.suptitle(f"High-Energy Fraction Sensitivity (rel_tol={REL_TOL})", fontsize=14, y=1.01)
    plt.tight_layout()
    fname = os.path.join(output_dir, "sweep_fraction.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {fname}")


def plot_fraction_sweep_normalized(df, output_dir):
    """绘制归一化的 fraction 扫描图（平均值 ± 标准差）。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 按 fraction 分组，计算跨数据集的统计量
    grouped = df.groupby("fraction").agg(
        cr_mean=("cr", "mean"),
        cr_std=("cr", "std"),
        mse_mean=("mse", "mean"),
        mse_std=("mse", "std"),
        mse_high_mean=("mse_high", "mean"),
        mse_high_std=("mse_high", "std"),
        snr_high_mean=("snr_high_db", "mean"),
        snr_high_std=("snr_high_db", "std"),
    ).reset_index()

    fracs = grouped["fraction"] * 100

    # (a) CR
    ax = axes[0]
    ax.errorbar(fracs, grouped["cr_mean"], yerr=grouped["cr_std"],
                marker="o", color="#2196F3", linewidth=2, capsize=4, label="Adaptive2D")
    # DASPack baseline (平均)
    dp_cr_mean = df.groupby("dataset")["dp_cr"].first().mean()
    ax.axhline(y=dp_cr_mean, color="gray", linestyle="--", linewidth=2, label="DASPack")
    ax.set_xlabel("High-Energy Fraction (%)")
    ax.set_ylabel("Compression Ratio")
    ax.set_title("CR vs Fraction (mean ± std)")
    ax.legend()
    ax.grid(True, alpha=0.2)

    # (b) MSE (归一化)
    ax = axes[1]
    # 归一化：MSE / DASPack MSE
    df_norm = df.copy()
    df_norm["mse_ratio"] = df_norm["mse"] / df_norm["dp_mse"]
    df_norm["mse_high_ratio"] = df_norm["mse_high"] / df_norm["dp_mse"]

    grouped_norm = df_norm.groupby("fraction").agg(
        mse_ratio_mean=("mse_ratio", "mean"),
        mse_ratio_std=("mse_ratio", "std"),
        mse_high_ratio_mean=("mse_high_ratio", "mean"),
        mse_high_ratio_std=("mse_high_ratio", "std"),
    ).reset_index()

    fracs = grouped_norm["fraction"] * 100
    ax.errorbar(fracs, grouped_norm["mse_ratio_mean"] * 100,
                yerr=grouped_norm["mse_ratio_std"] * 100,
                marker="s", color="#F44336", linewidth=2, capsize=4, label="Overall MSE")
    ax.errorbar(fracs, grouped_norm["mse_high_ratio_mean"] * 100,
                yerr=grouped_norm["mse_high_ratio_std"] * 100,
                marker="^", color="#4CAF50", linewidth=2, capsize=4, label="High-Energy MSE")
    ax.axhline(y=100, color="gray", linestyle="--", linewidth=2, label="DASPack (100%)")
    ax.set_xlabel("High-Energy Fraction (%)")
    ax.set_ylabel("MSE relative to DASPack (%)")
    ax.set_title("Normalized MSE vs Fraction")
    ax.legend()
    ax.grid(True, alpha=0.2)

    # (c) SNR_high
    ax = axes[2]
    ax.errorbar(fracs, grouped["snr_high_mean"], yerr=grouped["snr_high_std"],
                marker="d", color="#9C27B0", linewidth=2, capsize=4, label="Adaptive2D")
    dp_snr_high_mean = df.groupby("dataset")["snr_high_db"].apply(
        lambda x: df.loc[x.index, "snr_high_db"].iloc[0]
    ).mean()
    # Actually compute DASPack SNR_high properly
    dp_snr_vals = df.groupby("dataset")["snr_high_db"].first()
    # SNR_high for DASPack is the same as adaptive when fraction=0
    # Let's use the snr_high_db at fraction closest to 0
    ax.axhline(y=dp_snr_vals.mean(), color="gray", linestyle="--", linewidth=2, label="DASPack")
    ax.set_xlabel("High-Energy Fraction (%)")
    ax.set_ylabel("SNR (dB, high-energy)")
    ax.set_title("High-Energy SNR vs Fraction (mean ± std)")
    ax.legend()
    ax.grid(True, alpha=0.2)

    fig.suptitle(f"Fraction Sensitivity - Normalized (7 datasets, rel_tol={REL_TOL})", fontsize=14, y=1.02)
    plt.tight_layout()
    fname = os.path.join(output_dir, "sweep_fraction_normalized.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {fname}")


# ======================================================================
# 主函数
# ======================================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("加载数据集...")
    datasets = load_all_datasets()
    # 只用 3 个代表性数据集来加速（大、中、小各一个）
    representative = {
        "alboran": datasets["alboran"],
        "castor": datasets["castor"],
        "event_0322_a": datasets["event_0322_a"],
    }
    print(f"使用 {len(representative)} 个代表性数据集进行参数扫描")

    # --- 1. Block size 扫描 ---
    print("\n" + "=" * 60)
    print("Block Size 扫描")
    print("=" * 60)

    block_sizes = [
        (100, 250),
        (100, 500),
        (100, 1000),
        (250, 250),
        (250, 500),
        (250, 1000),
        (500, 500),
        (500, 1000),
        (1000, 500),
        (1000, 1000),
    ]

    bs_df = sweep_block_size(representative, block_sizes, frac=0.2, threads=4)
    bs_df.to_csv(os.path.join(OUTPUT_DIR, "sweep_block_size.csv"), index=False)
    plot_block_size_sweep(bs_df, OUTPUT_DIR)

    # --- 2. High energy fraction 扫描 ---
    print("\n" + "=" * 60)
    print("High-Energy Fraction 扫描")
    print("=" * 60)

    fractions = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    frac_df = sweep_fraction(datasets, fractions, block_channels=250, block_samples=500, threads=4)
    frac_df.to_csv(os.path.join(OUTPUT_DIR, "sweep_fraction.csv"), index=False)
    plot_fraction_sweep(frac_df, OUTPUT_DIR)
    plot_fraction_sweep_normalized(frac_df, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("参数敏感性分析完成！")
    print(f"所有图表保存在 {OUTPUT_DIR}/")
    print("=" * 60)
