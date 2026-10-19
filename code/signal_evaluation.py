"""
信号质量评估：波形对比、误差分布、块能量分布可视化。

对比 DASPack（step=2ε）和 Adaptive2D 在相同误差上界 ε 下的信号质量，
重点展示高能量区域（地震事件）的保真度差异。

用法:
  cd /home/xdzhang/DAS_Compression/DASPack/improve/daspack-main/benchmark
  python signal_evaluation.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tempfile import NamedTemporaryFile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fair_benchmark import (
    load_hdas_segment,
    DASPackFair,
    FairOptions,
    compute_metrics,
)
from adaptive_daspack_2d import AdaptiveDASPack2D as AD2D


def compress_and_restore(codec, arr, opts):
    """压缩并解压，返回重建数据。"""
    tmp = NamedTemporaryFile(suffix=".h5", delete=False)
    tmp_name = tmp.name
    tmp.close()
    try:
        codec.write(arr, tmp_name, opts)
        restored = codec.read(tmp_name)
    finally:
        os.remove(tmp_name)
    return restored


def plot_waveform_comparison(original, daspack_restored, adaptive_restored,
                             block_map, channel_idx, time_range, epsilon,
                             ds_name, output_dir="."):
    """
    绘制波形对比图：原始 vs DASPack vs Adaptive2D。

    Parameters
    ----------
    channel_idx : int
        要展示的通道索引。
    time_range : (int, int)
        时间样本范围。
    """
    t0, t1 = time_range
    orig = original[channel_idx, t0:t1]
    dp = daspack_restored[channel_idx, t0:t1]
    ad = adaptive_restored[channel_idx, t0:t1]

    # 判断该通道在该时间段是否属于高能量块
    is_high = block_map[channel_idx, (t0 + t1) // 2] == 1

    fig, axes = plt.subplots(2, 2, figsize=(16, 8))

    # 波形对比
    ax = axes[0, 0]
    ax.plot(orig, color="black", alpha=0.7, linewidth=0.8, label="Original")
    ax.plot(dp, color="blue", alpha=0.6, linewidth=0.8, label="DASPack")
    ax.plot(ad, color="red", alpha=0.6, linewidth=0.8, label="Adaptive2D")
    ax.set_xlabel("Time sample")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Ch{channel_idx} waveform (ε={epsilon}, {'HIGH' if is_high else 'LOW'} energy)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 误差对比
    ax = axes[0, 1]
    err_dp = dp - orig
    err_ad = ad - orig
    ax.plot(err_dp, color="blue", alpha=0.6, linewidth=0.8, label=f"DASPack (max={np.max(np.abs(err_dp)):.4f})")
    ax.plot(err_ad, color="red", alpha=0.6, linewidth=0.8, label=f"Adaptive2D (max={np.max(np.abs(err_ad)):.4f})")
    ax.axhline(epsilon, color="gray", linestyle="--", alpha=0.5, label=f"±ε=±{epsilon}")
    ax.axhline(-epsilon, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time sample")
    ax.set_ylabel("Error")
    ax.set_title("Reconstruction error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 频谱对比
    ax = axes[1, 0]
    freq_orig = np.abs(np.fft.rfft(orig))
    freq_dp = np.abs(np.fft.rfft(dp))
    freq_ad = np.abs(np.fft.rfft(ad))
    ax.plot(freq_orig, color="black", alpha=0.7, linewidth=0.8, label="Original")
    ax.plot(freq_dp, color="blue", alpha=0.5, linewidth=0.8, label="DASPack")
    ax.plot(freq_ad, color="red", alpha=0.5, linewidth=0.8, label="Adaptive2D")
    ax.set_xlabel("Frequency bin")
    ax.set_ylabel("Magnitude")
    ax.set_title("Frequency spectrum")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 误差直方图
    ax = axes[1, 1]
    ax.hist(err_dp, bins=50, color="blue", alpha=0.5, density=True, label="DASPack")
    ax.hist(err_ad, bins=50, color="red", alpha=0.5, density=True, label="Adaptive2D")
    ax.axvline(epsilon, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(-epsilon, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Error")
    ax.set_ylabel("Density")
    ax.set_title("Error distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"{ds_name} - Channel {channel_idx} (ε={epsilon})", fontsize=14, y=1.02)
    plt.tight_layout()
    fname = os.path.join(output_dir, f"waveform_{ds_name}_ch{channel_idx}_eps{epsilon}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {fname}")


def plot_error_map(original, daspack_restored, adaptive_restored,
                   block_map, epsilon, ds_name, output_dir="."):
    """绘制 2D 误差地图和块分类图。"""
    err_dp = np.abs(daspack_restored - original)
    err_ad = np.abs(adaptive_restored - original)

    # 降采样以便可视化（取每 N 个点的最大值）
    step_ch = max(1, err_dp.shape[0] // 500)
    step_t = max(1, err_dp.shape[1] // 500)
    err_dp_ds = err_dp[::step_ch, ::step_t]
    err_ad_ds = err_ad[::step_ch, ::step_t]
    block_ds = block_map[::step_ch, ::step_t]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # DASPack 误差
    im0 = axes[0].imshow(err_dp_ds, aspect="auto", cmap="hot", vmin=0, vmax=epsilon)
    axes[0].set_title(f"DASPack |error| (ε={epsilon})")
    axes[0].set_xlabel("Time sample")
    axes[0].set_ylabel("Channel")
    plt.colorbar(im0, ax=axes[0])

    # Adaptive2D 误差
    im1 = axes[1].imshow(err_ad_ds, aspect="auto", cmap="hot", vmin=0, vmax=epsilon)
    axes[1].set_title(f"Adaptive2D |error| (ε={epsilon})")
    axes[1].set_xlabel("Time sample")
    axes[1].set_ylabel("Channel")
    plt.colorbar(im1, ax=axes[1])

    # 块分类图
    im2 = axes[2].imshow(block_ds, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)
    axes[2].set_title("Block classification (red=high energy)")
    axes[2].set_xlabel("Time sample")
    axes[2].set_ylabel("Channel")
    plt.colorbar(im2, ax=axes[2])

    plt.suptitle(f"{ds_name} - Error map & block classification", fontsize=14, y=1.02)
    plt.tight_layout()
    fname = os.path.join(output_dir, f"error_map_{ds_name}_eps{epsilon}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {fname}")


def plot_block_energy_distribution(arr, block_channels=500, block_samples=500,
                                   threshold_factor=1.5, ds_name="", output_dir="."):
    """绘制块能量分布直方图。"""
    channels, samples = arr.shape
    global_std = np.std(arr)
    global_mean = np.mean(arr)

    energies = []
    types = []

    for ch_start in range(0, channels, block_channels):
        for t_start in range(0, samples, block_samples):
            ch_end = min(ch_start + block_channels, channels)
            t_end = min(t_start + block_samples, samples)
            block = arr[ch_start:ch_end, t_start:t_end].astype(np.float64)
            e = float(np.sqrt(np.mean(block**2)))
            energies.append(e)
            types.append("high" if e > threshold_factor * global_std else "low")

    energies = np.array(energies)
    threshold = threshold_factor * global_std

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 能量分布
    ax = axes[0]
    high_mask = np.array([t == "high" for t in types])
    low_mask = ~high_mask

    if low_mask.any():
        ax.hist(energies[low_mask], bins=30, color="blue", alpha=0.6, label=f"Low energy ({low_mask.sum()})")
    if high_mask.any():
        ax.hist(energies[high_mask], bins=30, color="red", alpha=0.6, label=f"High energy ({high_mask.sum()})")
    ax.axvline(threshold, color="green", linestyle="--", linewidth=2,
               label=f"Threshold (τσ={threshold:.2f})")
    ax.axvline(global_std, color="orange", linestyle=":", linewidth=2,
               label=f"σ={global_std:.2f}")
    ax.set_xlabel("Block RMS Energy")
    ax.set_ylabel("Count")
    ax.set_title(f"{ds_name} - Block energy distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 归一化能量分布
    ax = axes[1]
    normalized = energies / global_std if global_std > 0 else energies
    if low_mask.any():
        ax.hist(normalized[low_mask], bins=30, color="blue", alpha=0.6, label=f"Low energy ({low_mask.sum()})")
    if high_mask.any():
        ax.hist(normalized[high_mask], bins=30, color="red", alpha=0.6, label=f"High energy ({high_mask.sum()})")
    ax.axvline(threshold_factor, color="green", linestyle="--", linewidth=2,
               label=f"τ={threshold_factor}")
    ax.set_xlabel("Normalized Energy (E/σ)")
    ax.set_ylabel("Count")
    ax.set_title(f"{ds_name} - Normalized energy distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(output_dir, f"block_energy_{ds_name}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {fname}")

    # 打印统计
    print(f"\n  {ds_name} 块能量统计:")
    print(f"    σ = {global_std:.4f}")
    print(f"    块数: {len(energies)} (高能量: {high_mask.sum()}, 低能量: {low_mask.sum()})")
    print(f"    能量范围: [{energies.min():.4f}, {energies.max():.4f}]")
    print(f"    归一化范围: [{normalized.min():.4f}, {normalized.max():.4f}]")
    print(f"    阈值 τσ = {threshold:.4f} (τ={threshold_factor})")


if __name__ == "__main__":
    data_dir = "./data"

    datasets = {
        "alboran": load_hdas_segment(os.path.join(data_dir, "7M_S_20231005-214004_RAW.h5"))[0],
        "castor": load_hdas_segment(os.path.join(data_dir, "ZI_C_20230609-102630_RAW.h5"))[0],
    }

    epsilon = 0.1
    block_channels = 500
    block_samples = 500
    threshold_factor = 1.5

    daspack = DASPackFair(threads=4)
    adaptive = AD2D(
        tolerance=epsilon,
        block_channels=block_channels,
        block_samples=block_samples,
        threshold_factor=threshold_factor,
        threads=4,
    )
    opts = FairOptions(epsilon=epsilon)

    for ds_name, arr in datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        # 块能量分布
        plot_block_energy_distribution(
            arr, block_channels, block_samples, threshold_factor, ds_name
        )

        # 压缩并解压
        print(f"  压缩 DASPack (ε={epsilon})...")
        dp_restored = compress_and_restore(daspack, arr, opts)

        print(f"  压缩 Adaptive2D (ε={epsilon})...")
        ad_restored = compress_and_restore(adaptive, arr, opts)

        # 块分类图
        block_map, block_info = adaptive.get_block_map(arr, opts)

        # 误差地图
        plot_error_map(arr, dp_restored, ad_restored, block_map, epsilon, ds_name)

        # 波形对比：选择高能量区域的通道
        n_high = sum(1 for b in block_info if b["is_high"])
        if n_high > 0:
            # 找一个高能量块的通道
            for b in block_info:
                if b["is_high"]:
                    ch_mid = (b["ch_range"][0] + b["ch_range"][1]) // 2
                    t_mid = (b["t_range"][0] + b["t_range"][1]) // 2
                    t_half = 200
                    t0 = max(0, t_mid - t_half)
                    t1 = min(arr.shape[1], t_mid + t_half)
                    plot_waveform_comparison(
                        arr, dp_restored, ad_restored, block_map,
                        ch_mid, (t0, t1), epsilon, ds_name
                    )
                    break
        else:
            print(f"  (无高能量块，选择能量最高的通道)")
            # 找能量最大的块
            max_block = max(block_info, key=lambda b: b["energy"])
            ch_mid = (max_block["ch_range"][0] + max_block["ch_range"][1]) // 2
            t_mid = (max_block["t_range"][0] + max_block["t_range"][1]) // 2
            t_half = 200
            t0 = max(0, t_mid - t_half)
            t1 = min(arr.shape[1], t_mid + t_half)
            plot_waveform_comparison(
                arr, dp_restored, ad_restored, block_map,
                ch_mid, (t0, t1), epsilon, ds_name
            )

        # 波形对比：低能量区域
        for b in block_info:
            if not b["is_high"]:
                ch_mid = (b["ch_range"][0] + b["ch_range"][1]) // 2
                t_mid = (b["t_range"][0] + b["t_range"][1]) // 2
                t_half = 200
                t0 = max(0, t_mid - t_half)
                t1 = min(arr.shape[1], t_mid + t_half)
                plot_waveform_comparison(
                    arr, dp_restored, ad_restored, block_map,
                    ch_mid, (t0, t1), epsilon, ds_name + "_low"
                )
                break

    print("\n信号质量评估完成。")
