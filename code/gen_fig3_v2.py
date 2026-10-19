#!/usr/bin/env python3
"""
Generate new Figure 3: 4-panel error map comparison (pure Python version).

Panels:
(a) Original data + block classification overlay
(b) DASPack error map (uniform quantization, step=2ε)
(c) Adaptive2D error map (block-adaptive quantization)
(d) Difference map showing improvement
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py

# ============ Configuration ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "improve/daspack-main/benchmark/data/7M_S_20231005-214004_RAW.h5")
OUTPUT = os.path.join(BASE_DIR, "figures/fig3_error_map.png")
EPSILON_REL = 0.001
BLOCK_CH = 250
BLOCK_T = 500
HIGH_FRAC = 0.2

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
})

# ============ Pure Python quantization ============
def uniform_quantize(arr, step):
    """Uniform quantization with given step size. Returns quantized + dequantized."""
    # Use float64 to avoid overflow
    q = np.round(arr.astype(np.float64) / step)
    restored = q * step
    return restored

def compute_block_energies(data, block_ch, block_t):
    """Compute RMS energy for each 2D block, covering ALL data including partial blocks."""
    n_ch, n_t = data.shape
    n_blocks_ch = (n_ch + block_ch - 1) // block_ch  # ceiling division
    n_blocks_t = (n_t + block_t - 1) // block_t
    
    energies = np.zeros((n_blocks_ch, n_blocks_t))
    block_info = []
    
    for i in range(n_blocks_ch):
        for j in range(n_blocks_t):
            ch_start = i * block_ch
            ch_end = min((i + 1) * block_ch, n_ch)
            t_start = j * block_t
            t_end = min((j + 1) * block_t, n_t)
            
            block = data[ch_start:ch_end, t_start:t_end]
            energy = np.sqrt(np.mean(block ** 2))
            energies[i, j] = energy
            block_info.append({
                'i': i, 'j': j,
                'ch_range': (ch_start, ch_end),
                't_range': (t_start, t_end),
                'energy': energy
            })
    
    return energies, block_info

def classify_blocks(block_info, high_frac):
    """Classify blocks as high/low energy using percentile threshold."""
    energies = np.array([b['energy'] for b in block_info])
    threshold = np.percentile(energies, 100 * (1 - high_frac))
    
    for b in block_info:
        b['is_high'] = b['energy'] >= threshold
    
    return block_info

def build_block_map(block_info, shape):
    """Build a 2D map of block classifications."""
    block_map = np.zeros(shape, dtype=np.float32)
    for b in block_info:
        ch_start, ch_end = b['ch_range']
        t_start, t_end = b['t_range']
        if b['is_high']:
            block_map[ch_start:ch_end, t_start:t_end] = 1.0
    return block_map

def adaptive_quantize(data, block_info, epsilon, block_ch, block_t):
    """Adaptive quantization: step=ε for high-energy, step=2ε for low-energy."""
    restored = np.zeros_like(data)
    
    step_high = epsilon       # max error = step/2 = ε/2 (well within bound)
    step_low = 2.0 * epsilon  # max error = step/2 = ε (at bound)
    
    for b in block_info:
        ch_start, ch_end = b['ch_range']
        t_start, t_end = b['t_range']
        block_data = data[ch_start:ch_end, t_start:t_end]
        
        if b['is_high']:
            restored[ch_start:ch_end, t_start:t_end] = uniform_quantize(block_data, step_high)
        else:
            restored[ch_start:ch_end, t_start:t_end] = uniform_quantize(block_data, step_low)
    
    return restored

# ============ Load data ============
print("Loading Alboran data...")
with h5py.File(DATA_FILE, "r") as f:
    arr = np.array(f["data"][:]).astype(np.float64)
print(f"Shape: {arr.shape}, dtype={arr.dtype}, min={arr.min():.4f}, max={arr.max():.4f}")

data_std = float(np.std(arr))
epsilon = EPSILON_REL * data_std
print(f"std={data_std:.4f}, epsilon={epsilon:.4f}")

# ============ Compress/restore ============
print("Computing block energies...")
energies, block_info = compute_block_energies(arr, BLOCK_CH, BLOCK_T)
print(f"Blocks: {len(block_info)} ({energies.shape[0]}x{energies.shape[1]})")

print("Classifying blocks...")
block_info = classify_blocks(block_info, HIGH_FRAC)
n_high = sum(1 for b in block_info if b['is_high'])
n_low = sum(1 for b in block_info if not b['is_high'])
print(f"High-energy: {n_high}, Low-energy: {n_low}")

print("Building block map...")
block_map = build_block_map(block_info, arr.shape)

print("DASPack compression (uniform, step=2ε)...")
dp_restored = uniform_quantize(arr, 2.0 * epsilon)

print("Adaptive2D compression...")
ad_restored = adaptive_quantize(arr, block_info, epsilon, BLOCK_CH, BLOCK_T)

# Compute errors
err_dp = np.abs(dp_restored - arr)
err_ad = np.abs(ad_restored - arr)
err_diff = err_dp - err_ad  # positive = adaptive is better

# Statistics
mse_dp = np.mean(err_dp ** 2)
mse_ad = np.mean(err_ad ** 2)
high_mask = block_map == 1
mse_dp_high = np.mean(err_dp[high_mask] ** 2) if np.any(high_mask) else 0
mse_ad_high = np.mean(err_ad[high_mask] ** 2) if np.any(high_mask) else 0
print(f"\nDASPack MSE: {mse_dp:.6f}, MSE_high: {mse_dp_high:.6f}")
print(f"Adaptive2D MSE: {mse_ad:.6f}, MSE_high: {mse_ad_high:.6f}")
print(f"High-energy improvement: {((mse_dp_high - mse_ad_high)/mse_dp_high*100):.1f}%")

# ============ Generate 4-panel figure ============
print("\nGenerating figure...")

# Downsample for display
step_ch = max(1, arr.shape[0] // 300)
step_t = max(1, arr.shape[1] // 300)
orig_ds = arr[::step_ch, ::step_t]
err_dp_ds = err_dp[::step_ch, ::step_t]
err_ad_ds = err_ad[::step_ch, ::step_t]
err_diff_ds = err_diff[::step_ch, ::step_t]
block_ds = block_map[::step_ch, ::step_t]

# Figure: 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

vmax_err = max(epsilon * 1.1, np.percentile(err_dp_ds, 99))

# (a) Original data + block classification
ax = axes[0, 0]
vmax_orig = max(abs(np.percentile(orig_ds, 1)), abs(np.percentile(orig_ds, 99)))
im_a = ax.imshow(orig_ds, aspect="auto", cmap="seismic",
                 vmin=-vmax_orig, vmax=vmax_orig)
block_overlay = np.ma.masked_where(block_ds == 0, block_ds)
ax.imshow(block_overlay, aspect="auto", cmap="YlOrRd", alpha=0.4, vmin=0, vmax=1)
ax.contour(block_ds, levels=[0.5], colors="black", linewidths=0.6, alpha=0.6)
ax.set_title("(a) Original data with block classification", fontsize=11, fontweight="bold")
ax.set_xlabel("Time sample")
ax.set_ylabel("Channel")
plt.colorbar(im_a, ax=ax, label="Amplitude", shrink=0.85)

# (b) DASPack error
ax = axes[0, 1]
im_b = ax.imshow(err_dp_ds, aspect="auto", cmap="inferno", vmin=0, vmax=vmax_err)
ax.contour(block_ds, levels=[0.5], colors="white", linewidths=0.8, alpha=0.5, linestyles="--")
ax.set_title("(b) DASPack |error|", fontsize=11, fontweight="bold")
ax.set_xlabel("Time sample")
ax.set_ylabel("Channel")
plt.colorbar(im_b, ax=ax, label="|error|", shrink=0.85)

# (c) Adaptive2D error
ax = axes[1, 0]
im_c = ax.imshow(err_ad_ds, aspect="auto", cmap="inferno", vmin=0, vmax=vmax_err)
ax.contour(block_ds, levels=[0.5], colors="white", linewidths=0.8, alpha=0.5, linestyles="--")
ax.set_title("(c) Adaptive2D |error|", fontsize=11, fontweight="bold")
ax.set_xlabel("Time sample")
ax.set_ylabel("Channel")
plt.colorbar(im_c, ax=ax, label="|error|", shrink=0.85)

# (d) Difference map (DASPack - Adaptive2D)
ax = axes[1, 1]
vmax_diff = max(abs(np.percentile(err_diff_ds, 1)), abs(np.percentile(err_diff_ds, 99)))
vmax_diff = max(vmax_diff, epsilon * 0.3)
im_d = ax.imshow(err_diff_ds, aspect="auto", cmap="RdBu_r",
                 vmin=-vmax_diff, vmax=vmax_diff)
ax.contour(block_ds, levels=[0.5], colors="black", linewidths=0.8, alpha=0.5, linestyles="--")
ax.set_title("(d) Error reduction (DASPack $-$ Adaptive2D)", fontsize=11, fontweight="bold")
ax.set_xlabel("Time sample")
ax.set_ylabel("Channel")
cbar_d = plt.colorbar(im_d, ax=ax, label="Error difference", shrink=0.85)
cbar_d.ax.set_ylabel("Red = adaptive better", rotation=270, labelpad=12)

# Statistics annotation
if np.any(high_mask):
    high_avg_err_dp = np.mean(err_dp[high_mask])
    high_avg_err_ad = np.mean(err_ad[high_mask])
    high_avg_reduction = high_avg_err_dp - high_avg_err_ad
    high_reduction_pct = high_avg_reduction / high_avg_err_dp * 100
else:
    high_avg_err_dp = high_avg_err_ad = high_avg_reduction = high_reduction_pct = 0

low_mask = block_map == 0
if np.any(low_mask):
    low_avg_reduction = np.mean(err_diff[low_mask])
else:
    low_avg_reduction = 0

stats_text = (f"High-energy blocks: {high_avg_err_dp:.4f} -> {high_avg_err_ad:.4f}\n"
              f"  (reduction: {high_avg_reduction:.4f}, {high_reduction_pct:.1f}%)\n"
              f"Low-energy blocks: reduction = {low_avg_reduction:.6f}")
ax.annotate(stats_text, xy=(0.02, 0.02), xycoords="axes fraction",
            fontsize=8.5, va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                     edgecolor="#555", alpha=0.92))

fig.suptitle(
    f"Alboran dataset $-$ Error comparison ($\\varepsilon$ = {epsilon:.3f})",
    fontsize=13, fontweight="bold", y=1.00
)
plt.tight_layout(pad=1.0, h_pad=1.2, w_pad=1.0)
fig.savefig(OUTPUT, bbox_inches="tight", facecolor="white", edgecolor="none")
plt.close()
print(f"\nSaved to {OUTPUT}")
print(f"\nHigh-energy: DASPack error={high_avg_err_dp:.4f}, Adaptive2D error={high_avg_err_ad:.4f}")
print(f"  Improvement: {high_avg_reduction:.4f} ({high_reduction_pct:.1f}%)")
