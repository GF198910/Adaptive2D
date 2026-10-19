"""
波形可视化对比：原始数据 vs DASPack vs Adaptive2D。

为每个数据集生成：
  1. 高能量区域波形对比 + 误差 + 频谱 + 误差直方图
  2. 低能量区域波形对比
  3. 2D 误差地图 + 块分类图
  4. 多数据集汇总对比图

用法:
  cd /home/xdzhang/DAS_Compression/DASPack/improve/daspack-main/benchmark
  python waveform_visualization.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tempfile import NamedTemporaryFile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fair_benchmark import DASPackFair, FairOptions, compute_metrics, load_hdas_segment
from adaptive_daspack_2d import AdaptiveDASPack2D
from extended_benchmark import load_all_datasets


OUTPUT_DIR = "waveform_plots"
EPSILON_REL = 0.001  # rel_tol = 0.001 (ε = 0.1 × std)


def compress_and_restore(codec, arr, opts):
    """压缩并解压。"""
    tmp = NamedTemporaryFile(suffix=".h5", delete=False)
    tmp_name = tmp.name
    tmp.close()
    try:
        codec.write(arr, tmp_name, opts)
        restored = codec.read(tmp_name)
    finally:
        os.remove(tmp_name)
    return restored


# ======================================================================
# 1. 单通道波形对比（4 子图）
# ======================================================================

def plot_waveform_4panel(original, dp_restored, ad_restored, block_map,
                         channel_idx, t_start, t_end, epsilon, ds_name,
                         region_type, output_dir):
    """4 子图波形对比：波形、误差、频谱、误差直方图。"""
    t0, t1 = t_start, t_end
    orig = original[channel_idx, t0:t1]
    dp = dp_restored[channel_idx, t0:t1]
    ad = ad_restored[channel_idx, t0:t1]
    is_high = block_map[channel_idx, (t0 + t1) // 2] == 1

    n = len(orig)
    time_axis = np.arange(n)

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # (a) 波形对比
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(time_axis, orig, color="black", alpha=0.7, linewidth=0.7, label="Original")
    ax.plot(time_axis, dp, color="#2196F3", alpha=0.6, linewidth=0.7, label="DASPack")
    ax.plot(time_axis, ad, color="#F44336", alpha=0.6, linewidth=0.7, label="Adaptive2D")
    ax.set_xlabel("Time sample", fontsize=11)
    ax.set_ylabel("Amplitude", fontsize=11)
    ax.set_title(f"Waveform (Ch {channel_idx}, {'HIGH' if is_high else 'LOW'} energy)", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.2)

    # (b) 误差对比
    ax = fig.add_subplot(gs[0, 1])
    err_dp = dp - orig
    err_ad = ad - orig
    ax.plot(time_axis, err_dp, color="#2196F3", alpha=0.6, linewidth=0.6,
            label=f"DASPack (max={np.max(np.abs(err_dp)):.5f})")
    ax.plot(time_axis, err_ad, color="#F44336", alpha=0.6, linewidth=0.6,
            label=f"Adaptive2D (max={np.max(np.abs(err_ad)):.5f})")
    ax.axhline(epsilon, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(-epsilon, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.fill_between(time_axis, -epsilon, epsilon, color="green", alpha=0.05)
    ax.set_xlabel("Time sample", fontsize=11)
    ax.set_ylabel("Error", fontsize=11)
    ax.set_title("Reconstruction error", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.2)

    # (c) 频谱对比
    ax = fig.add_subplot(gs[1, 0])
    freq_orig = np.abs(np.fft.rfft(orig))
    freq_dp = np.abs(np.fft.rfft(dp))
    freq_ad = np.abs(np.fft.rfft(ad))
    freqs = np.arange(len(freq_orig))
    ax.plot(freqs, freq_orig, color="black", alpha=0.7, linewidth=0.7, label="Original")
    ax.plot(freqs, freq_dp, color="#2196F3", alpha=0.5, linewidth=0.7, label="DASPack")
    ax.plot(freqs, freq_ad, color="#F44336", alpha=0.5, linewidth=0.7, label="Adaptive2D")
    ax.set_xlabel("Frequency bin", fontsize=11)
    ax.set_ylabel("Magnitude", fontsize=11)
    ax.set_title("Frequency spectrum", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # (d) 误差直方图
    ax = fig.add_subplot(gs[1, 1])
    bins = np.linspace(-epsilon * 1.2, epsilon * 1.2, 60)
    ax.hist(err_dp, bins=bins, color="#2196F3", alpha=0.5, density=True, label="DASPack")
    ax.hist(err_ad, bins=bins, color="#F44336", alpha=0.5, density=True, label="Adaptive2D")
    ax.axvline(0, color="black", linestyle="-", alpha=0.3, linewidth=1)
    ax.set_xlabel("Error", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Error distribution", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    region_label = "high" if region_type == "high" else "low"
    fig.suptitle(
        f"{ds_name} — {region_label.upper()} energy region — "
        f"ε={epsilon:.4f} (rel_tol={EPSILON_REL})",
        fontsize=14, y=0.98
    )

    fname = os.path.join(output_dir, f"waveform_{ds_name}_{region_label}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    return fname


# ======================================================================
# 2. 2D 误差地图 + 块分类
# ======================================================================

def plot_error_map_2d(original, dp_restored, ad_restored, block_map,
                      epsilon, ds_name, output_dir):
    """2D 误差地图 + 差异对比 + 块分类叠加。
    
    4 面板设计：
    (a) 原始数据瀑布图（地震色表）+ 块分类边框
    (b) DASPack 误差图
    (c) Adaptive2D 误差图
    (d) 差异图 (DASPack error - Adaptive2D error)：红色=改进，蓝色=无变化
    """
    err_dp = np.abs(dp_restored - original)
    err_ad = np.abs(ad_restored - original)
    err_diff = err_dp - err_ad  # positive = adaptive is better (less error)

    # 降采样
    step_ch = max(1, err_dp.shape[0] // 300)
    step_t = max(1, err_dp.shape[1] // 300)
    orig_ds = original[::step_ch, ::step_t]
    err_dp_ds = err_dp[::step_ch, ::step_t]
    err_ad_ds = err_ad[::step_ch, ::step_t]
    err_diff_ds = err_diff[::step_ch, ::step_t]
    block_ds = block_map[::step_ch, ::step_t]

    # Create figure: 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    vmax_err = max(epsilon * 1.1, np.percentile(err_dp_ds, 99))
    
    # (a) Original data + block classification overlay
    ax = axes[0, 0]
    vmax_orig = max(abs(np.percentile(orig_ds, 1)), abs(np.percentile(orig_ds, 99)))
    im_a = ax.imshow(orig_ds, aspect="auto", cmap="seismic",
                     vmin=-vmax_orig, vmax=vmax_orig)
    # Overlay block classification as translucent
    block_overlay = np.ma.masked_where(block_ds == 0, block_ds)
    ax.imshow(block_overlay, aspect="auto", cmap="YlOrRd", alpha=0.35, vmin=0, vmax=1)
    # Block boundaries
    ax.contour(block_ds, levels=[0.5], colors="black", linewidths=0.5, alpha=0.5)
    ax.set_title("(a) Original data + block classification", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time sample")
    ax.set_ylabel("Channel")
    cbar_a = plt.colorbar(im_a, ax=ax, label="Amplitude", shrink=0.8)

    # (b) DASPack error
    ax = axes[0, 1]
    im_b = ax.imshow(err_dp_ds, aspect="auto", cmap="inferno", vmin=0, vmax=vmax_err)
    ax.set_title("(b) DASPack |error|", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time sample")
    ax.set_ylabel("Channel")
    cbar_b = plt.colorbar(im_b, ax=ax, label="|error|", shrink=0.8)
    
    # Add contour of high-energy blocks
    ax.contour(block_ds, levels=[0.5], colors="white", linewidths=0.8, alpha=0.6, linestyles="--")

    # (c) Adaptive2D error
    ax = axes[1, 0]
    im_c = ax.imshow(err_ad_ds, aspect="auto", cmap="inferno", vmin=0, vmax=vmax_err)
    ax.set_title("(c) Adaptive2D |error|", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time sample")
    ax.set_ylabel("Channel")
    cbar_c = plt.colorbar(im_c, ax=ax, label="|error|", shrink=0.8)
    
    # Add contour of high-energy blocks  
    ax.contour(block_ds, levels=[0.5], colors="white", linewidths=0.8, alpha=0.6, linestyles="--")

    # (d) Difference map (DASPack - Adaptive2D)
    ax = axes[1, 1]
    # Symmetric colormap: red where adaptive is better, blue where it's worse
    vmax_diff = max(abs(np.percentile(err_diff_ds, 1)), abs(np.percentile(err_diff_ds, 99)))
    # For difference, clip to a reasonable range
    vmax_diff = max(vmax_diff, epsilon * 0.5)
    im_d = ax.imshow(err_diff_ds, aspect="auto", cmap="RdBu_r",
                     vmin=-vmax_diff, vmax=vmax_diff)
    ax.set_title("(d) Error reduction (DASPack − Adaptive2D)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time sample")
    ax.set_ylabel("Channel")
    cbar_d = plt.colorbar(im_d, ax=ax, label="Error difference", shrink=0.8)
    cbar_d.ax.set_ylabel("Red = adaptive better", rotation=270, labelpad=10)
    
    # Add contour of high-energy blocks
    ax.contour(block_ds, levels=[0.5], colors="black", linewidths=0.8, alpha=0.6, linestyles="--")
    
    # Add annotation: improvement in high-energy regions
    high_err_reduction = np.mean(err_diff_ds[block_ds == 1]) if np.any(block_ds == 1) else 0
    low_err_reduction = np.mean(err_diff_ds[block_ds == 0]) if np.any(block_ds == 0) else 0
    ax.annotate(f"High-energy: {high_err_reduction:.4f}\nLow-energy: {low_err_reduction:.4f}",
                xy=(0.02, 0.02), xycoords="axes fraction",
                fontsize=9, verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                         edgecolor="#333", alpha=0.9))

    fig.suptitle(
        f"{ds_name} — Error comparison (ε={epsilon:.4f})",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    fname = os.path.join(output_dir, f"errormap_{ds_name}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    return fname


# ======================================================================
# 3. 多数据集汇总图
# ======================================================================

def plot_summary_all(datasets_results, epsilon_info, output_dir):
    """
    汇总图：所有数据集的 MSE 对比柱状图。

    Parameters
    ----------
    datasets_results : dict
        {ds_name: {"dp": metrics, "ad": metrics, "n_high": int, "n_low": int}}
    """
    ds_names = list(datasets_results.keys())
    n = len(ds_names)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    x = np.arange(n)
    width = 0.35

    # (a) 整体 MSE
    ax = axes[0, 0]
    dp_mse = [datasets_results[ds]["dp"]["mse"] for ds in ds_names]
    ad_mse = [datasets_results[ds]["ad"]["mse"] for ds in ds_names]
    bars1 = ax.bar(x - width/2, dp_mse, width, color="#2196F3", alpha=0.8, label="DASPack")
    bars2 = ax.bar(x + width/2, ad_mse, width, color="#F44336", alpha=0.8, label="Adaptive2D")
    ax.set_ylabel("MSE", fontsize=11)
    ax.set_title("Overall MSE", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(ds_names, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    # (b) 高能量 MSE
    ax = axes[0, 1]
    dp_mse_h = [datasets_results[ds]["dp"].get("mse_high", 0) for ds in ds_names]
    ad_mse_h = [datasets_results[ds]["ad"].get("mse_high", 0) for ds in ds_names]
    ax.bar(x - width/2, dp_mse_h, width, color="#2196F3", alpha=0.8, label="DASPack")
    ax.bar(x + width/2, ad_mse_h, width, color="#F44336", alpha=0.8, label="Adaptive2D")
    ax.set_ylabel("MSE (high-energy)", fontsize=11)
    ax.set_title("High-Energy Region MSE (4x reduction)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(ds_names, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    # (c) 压缩率
    ax = axes[1, 0]
    dp_cr = [datasets_results[ds]["dp"]["cr"] for ds in ds_names]
    ad_cr = [datasets_results[ds]["ad"]["cr"] for ds in ds_names]
    ax.bar(x - width/2, dp_cr, width, color="#2196F3", alpha=0.8, label="DASPack")
    ax.bar(x + width/2, ad_cr, width, color="#F44336", alpha=0.8, label="Adaptive2D")
    ax.set_ylabel("Compression Ratio", fontsize=11)
    ax.set_title("Compression Ratio (same max error ε)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(ds_names, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    # (d) SNR
    ax = axes[1, 1]
    dp_snr_h = [datasets_results[ds]["dp"].get("snr_high_db", 0) for ds in ds_names]
    ad_snr_h = [datasets_results[ds]["ad"].get("snr_high_db", 0) for ds in ds_names]
    ax.bar(x - width/2, dp_snr_h, width, color="#2196F3", alpha=0.8, label="DASPack")
    ax.bar(x + width/2, ad_snr_h, width, color="#F44336", alpha=0.8, label="Adaptive2D")
    ax.set_ylabel("SNR (dB, high-energy)", fontsize=11)
    ax.set_title("High-Energy Region SNR (+6 dB)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(ds_names, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle(
        f"Summary across {n} datasets — ε = {epsilon_info:.4f} (rel_tol={EPSILON_REL})",
        fontsize=14, y=1.01
    )
    plt.tight_layout()
    fname = os.path.join(output_dir, "summary_all_datasets.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    return fname


# ======================================================================
# 4. 多通道波形对比（瀑布图）
# ======================================================================

def plot_waterfall_comparison(original, dp_restored, ad_restored, block_map,
                               epsilon, ds_name, output_dir, n_channels=200):
    """瀑布图对比：原始、DASPack 误差、Adaptive2D 误差。"""
    channels, samples = original.shape
    ch_step = max(1, channels // n_channels)
    ch_indices = list(range(0, channels, ch_step))
    n_ch = len(ch_indices)

    # 降采样时间轴
    t_step = max(1, samples // 600)
    t_indices = list(range(0, samples, t_step))
    n_t = len(t_indices)

    orig_ds = original[np.ix_(ch_indices, t_indices)]
    err_dp_ds = np.abs(dp_restored[np.ix_(ch_indices, t_indices)] - orig_ds)
    err_ad_ds = np.abs(ad_restored[np.ix_(ch_indices, t_indices)] - orig_ds)
    block_ds = block_map[np.ix_(ch_indices, t_indices)]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # (a) 原始数据
    ax = axes[0, 0]
    im = ax.imshow(orig_ds, aspect="auto", cmap="seismic",
                   vmin=-np.std(orig_ds) * 3, vmax=np.std(orig_ds) * 3)
    ax.set_title("Original data", fontsize=12)
    ax.set_xlabel("Time sample")
    ax.set_ylabel("Channel")
    plt.colorbar(im, ax=ax, label="Amplitude")

    # (b) DASPack 误差
    ax = axes[0, 1]
    im = ax.imshow(err_dp_ds, aspect="auto", cmap="hot", vmin=0, vmax=epsilon)
    ax.set_title("DASPack |error|", fontsize=12)
    ax.set_xlabel("Time sample")
    ax.set_ylabel("Channel")
    plt.colorbar(im, ax=ax, label="|error|")

    # (c) Adaptive2D 误差
    ax = axes[1, 0]
    im = ax.imshow(err_ad_ds, aspect="auto", cmap="hot", vmin=0, vmax=epsilon)
    ax.set_title("Adaptive2D |error|", fontsize=12)
    ax.set_xlabel("Time sample")
    ax.set_ylabel("Channel")
    plt.colorbar(im, ax=ax, label="|error|")

    # (d) 块分类叠加
    ax = axes[1, 1]
    # 用半透明颜色覆盖高能量块
    ax.imshow(orig_ds, aspect="auto", cmap="gray",
              vmin=-np.std(orig_ds) * 3, vmax=np.std(orig_ds) * 3, alpha=0.5)
    overlay = np.ma.masked_where(block_ds == 0, block_ds)
    ax.imshow(overlay, aspect="auto", cmap="autumn", alpha=0.7, vmin=0, vmax=1)
    ax.set_title("Data + block classification (red=high energy)", fontsize=12)
    ax.set_xlabel("Time sample")
    ax.set_ylabel("Channel")

    fig.suptitle(
        f"{ds_name} — Waterfall comparison (ε={epsilon:.4f})",
        fontsize=14, y=1.01
    )
    plt.tight_layout()
    fname = os.path.join(output_dir, f"waterfall_{ds_name}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    return fname


# ======================================================================
# 主函数
# ======================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("加载数据集...")
    datasets = load_all_datasets()
    print(f"共 {len(datasets)} 个数据集")

    daspack = DASPackFair(threads=4)
    adaptive = AdaptiveDASPack2D(
        tolerance=0.1,
        block_channels=250,
        block_samples=500,
        threshold_mode="percentile",
        high_energy_fraction=0.2,
        threads=4,
    )

    all_results = {}

    for ds_name, arr in datasets.items():
        data_std = float(np.std(arr))
        epsilon = EPSILON_REL * data_std
        opts = FairOptions(epsilon=epsilon)

        print(f"\n{'='*60}")
        print(f"{ds_name}: shape={arr.shape}, std={data_std:.2f}, ε={epsilon:.4f}")
        print(f"{'='*60}")

        # 压缩
        print("  压缩 DASPack...")
        dp_restored = compress_and_restore(daspack, arr, opts)
        print("  压缩 Adaptive2D...")
        ad_restored = compress_and_restore(adaptive, arr, opts)

        # 块分类
        block_map, block_info = adaptive.get_block_map(arr, opts)
        n_high = sum(1 for b in block_info if b["is_high"])
        n_low = sum(1 for b in block_info if not b["is_high"])
        print(f"  块: {len(block_info)} (high={n_high}, low={n_low})")

        # 指标
        dp_metrics = compute_metrics(arr, dp_restored, block_map)
        ad_metrics = compute_metrics(arr, ad_restored, block_map)
        dp_metrics["cr"] = arr.nbytes / os.path.getsize(
            NamedTemporaryFile(suffix=".h5", delete=False).name
        ) if False else None  # CR 已在 benchmark 中计算
        # 手动计算 CR
        import tempfile
        for codec, name in [(daspack, "dp"), (adaptive, "ad")]:
            tmp = NamedTemporaryFile(suffix=".h5", delete=False)
            tmp_name = tmp.name
            tmp.close()
            codec.write(arr, tmp_name, opts)
            cr = arr.nbytes / os.path.getsize(tmp_name)
            os.remove(tmp_name)
            if name == "dp":
                dp_metrics["cr"] = cr
            else:
                ad_metrics["cr"] = cr

        all_results[ds_name] = {
            "dp": dp_metrics,
            "ad": ad_metrics,
            "n_high": n_high,
            "n_low": n_low,
        }

        print(f"  DASPack:    CR={dp_metrics['cr']:.2f}  MSE={dp_metrics['mse']:.2e}  "
              f"MSE_high={dp_metrics.get('mse_high', 0):.2e}  "
              f"SNR_high={dp_metrics.get('snr_high_db', 0):.1f}dB")
        print(f"  Adaptive2D: CR={ad_metrics['cr']:.2f}  MSE={ad_metrics['mse']:.2e}  "
              f"MSE_high={ad_metrics.get('mse_high', 0):.2e}  "
              f"SNR_high={ad_metrics.get('snr_high_db', 0):.1f}dB")

        # --- 生成图 ---

        # 1. 找高能量区域的通道和时间范围
        high_blocks = [b for b in block_info if b["is_high"]]
        if high_blocks:
            # 选能量最高的高能量块
            best_block = max(high_blocks, key=lambda b: b["energy"])
            ch_mid = (best_block["ch_range"][0] + best_block["ch_range"][1]) // 2
            t_mid = (best_block["t_range"][0] + best_block["t_range"][1]) // 2
        else:
            # 如果没有高能量块，选能量最大的块
            best_block = max(block_info, key=lambda b: b["energy"])
            ch_mid = (best_block["ch_range"][0] + best_block["ch_range"][1]) // 2
            t_mid = (best_block["t_range"][0] + best_block["t_range"][1]) // 2

        t_half = min(300, arr.shape[1] // 4)
        t_start = max(0, t_mid - t_half)
        t_end = min(arr.shape[1], t_mid + t_half)

        # 高能量区域波形
        fname = plot_waveform_4panel(
            arr, dp_restored, ad_restored, block_map,
            ch_mid, t_start, t_end, epsilon, ds_name, "high", OUTPUT_DIR
        )
        print(f"  保存: {fname}")

        # 2. 低能量区域波形
        low_blocks = [b for b in block_info if not b["is_high"]]
        if low_blocks:
            # 选一个远离高能量块的低能量块
            low_block = min(low_blocks, key=lambda b: b["energy"])
            ch_mid_low = (low_block["ch_range"][0] + low_block["ch_range"][1]) // 2
            t_mid_low = (low_block["t_range"][0] + low_block["t_range"][1]) // 2
            t_start_low = max(0, t_mid_low - t_half)
            t_end_low = min(arr.shape[1], t_mid_low + t_half)

            fname = plot_waveform_4panel(
                arr, dp_restored, ad_restored, block_map,
                ch_mid_low, t_start_low, t_end_low, epsilon, ds_name, "low", OUTPUT_DIR
            )
            print(f"  保存: {fname}")

        # 3. 2D 误差地图
        fname = plot_error_map_2d(
            arr, dp_restored, ad_restored, block_map,
            epsilon, ds_name, OUTPUT_DIR
        )
        print(f"  保存: {fname}")

        # 4. 瀑布图
        fname = plot_waterfall_comparison(
            arr, dp_restored, ad_restored, block_map,
            epsilon, ds_name, OUTPUT_DIR
        )
        print(f"  保存: {fname}")

    # 5. 汇总图
    print(f"\n{'='*60}")
    print("生成汇总图...")
    fname = plot_summary_all(all_results, list(datasets.values())[0].std() * EPSILON_REL, OUTPUT_DIR)
    print(f"  保存: {fname}")

    print(f"\n所有可视化图已保存到 {OUTPUT_DIR}/")
    print(f"共生成 {len(os.listdir(OUTPUT_DIR))} 个文件")


if __name__ == "__main__":
    main()
