#!/usr/bin/env python3
"""
Generate new Figure 2: Waveform comparison in DASPack-style layout.

Layout: 2 rows × 3 columns
Row 1 (High-energy channel):
  (a) Full waveform overview  (b) Zoomed detail  (c) Reconstruction error
Row 2 (Low-energy channel):
  (d) Full waveform overview  (e) Zoomed detail  (f) Reconstruction error
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
OUTPUT = os.path.join(BASE_DIR, "figures/fig2_waveform.png")
EPSILON_REL = 0.001
BLOCK_CH = 250
BLOCK_T = 500
HIGH_FRAC = 0.2

# Channels to display
HIGH_CH = 250    # High-energy channel
LOW_CH = 750     # Low-energy channel

# Time windows
FULL_T_START = 0
FULL_T_END = 6000
ZOOM_T_START = 2500
ZOOM_T_END = 3100

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#333333",
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linewidth": 0.4,
    "grid.linestyle": "--",
    "lines.linewidth": 0.8,
    "legend.fontsize": 8,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc",
})

# ============ Pure Python quantization ============
def uniform_quantize(arr, step):
    q = np.round(arr.astype(np.float64) / step)
    return q * step

def compute_block_energies(data, block_ch, block_t):
    n_ch, n_t = data.shape
    n_blocks_ch = (n_ch + block_ch - 1) // block_ch
    n_blocks_t = (n_t + block_t - 1) // block_t
    block_info = []
    for i in range(n_blocks_ch):
        for j in range(n_blocks_t):
            ch_s = i * block_ch
            ch_e = min((i + 1) * block_ch, n_ch)
            t_s = j * block_t
            t_e = min((j + 1) * block_t, n_t)
            block = data[ch_s:ch_e, t_s:t_e]
            energy = np.sqrt(np.mean(block ** 2))
            block_info.append({
                'ch_range': (ch_s, ch_e), 't_range': (t_s, t_e),
                'energy': energy, 'is_high': False
            })
    return block_info

def classify_blocks(block_info, high_frac):
    energies = np.array([b['energy'] for b in block_info])
    threshold = np.percentile(energies, 100 * (1 - high_frac))
    for b in block_info:
        b['is_high'] = b['energy'] >= threshold
    return block_info

def build_block_map(block_info, shape):
    block_map = np.zeros(shape, dtype=np.float32)
    for b in block_info:
        ch_s, ch_e = b['ch_range']
        t_s, t_e = b['t_range']
        if b['is_high']:
            block_map[ch_s:ch_e, t_s:t_e] = 1.0
    return block_map

def adaptive_quantize(data, block_info, epsilon):
    restored = np.zeros_like(data)
    for b in block_info:
        ch_s, ch_e = b['ch_range']
        t_s, t_e = b['t_range']
        block_data = data[ch_s:ch_e, t_s:t_e]
        if b['is_high']:
            restored[ch_s:ch_e, t_s:t_e] = uniform_quantize(block_data, epsilon)
        else:
            restored[ch_s:ch_e, t_s:t_e] = uniform_quantize(block_data, 2.0 * epsilon)
    return restored

# ============ Load data ============
print("Loading Alboran data...")
with h5py.File(DATA_FILE, "r") as f:
    arr = np.array(f["data"][:]).astype(np.float64)

data_std = float(np.std(arr))
epsilon = EPSILON_REL * data_std
print(f"Shape: {arr.shape}, std={data_std:.4f}, epsilon={epsilon:.4f}")

# ============ Compress ============
print("Computing blocks...")
block_info = compute_block_energies(arr, BLOCK_CH, BLOCK_T)
block_info = classify_blocks(block_info, HIGH_FRAC)
block_map = build_block_map(block_info, arr.shape)

print("DASPack compression...")
dp_restored = uniform_quantize(arr, 2.0 * epsilon)

print("Adaptive2D compression...")
ad_restored = adaptive_quantize(arr, block_info, epsilon)

# ============ Colors ============
C_ORIG = "#222222"
C_DP = "#2166ac"   # Blue
C_AD = "#b2182b"   # Red

# ============ Generate figure ============
print("Generating figure...")
fig, axes = plt.subplots(2, 3, figsize=(13, 5.5))

for row, (ch, label) in enumerate([(HIGH_CH, "High-energy"), (LOW_CH, "Low-energy")]):
    orig_full = arr[ch, FULL_T_START:FULL_T_END]
    dp_full = dp_restored[ch, FULL_T_START:FULL_T_END]
    ad_full = ad_restored[ch, FULL_T_START:FULL_T_END]
    
    orig_zoom = arr[ch, ZOOM_T_START:ZOOM_T_END]
    dp_zoom = dp_restored[ch, ZOOM_T_START:ZOOM_T_END]
    ad_zoom = ad_restored[ch, ZOOM_T_START:ZOOM_T_END]
    
    err_dp_full = dp_full - orig_full
    err_ad_full = ad_full - orig_full
    err_dp_zoom = dp_zoom - orig_zoom
    err_ad_zoom = ad_zoom - orig_zoom
    
    t_full = np.arange(FULL_T_START, FULL_T_END)
    t_zoom = np.arange(ZOOM_T_START, ZOOM_T_END)
    
    # (a/d) Full waveform overview
    ax = axes[row, 0]
    ax.plot(t_full, orig_full, color=C_ORIG, alpha=0.7, linewidth=0.5, label="Original")
    ax.plot(t_full, dp_full, color=C_DP, alpha=0.6, linewidth=0.4, label="DASPack")
    ax.plot(t_full, ad_full, color=C_AD, alpha=0.6, linewidth=0.4, label="Adaptive2D")
    # Mark zoom region
    ax.axvspan(ZOOM_T_START, ZOOM_T_END, alpha=0.08, color="gray")
    panel_letter = "(a)" if row == 0 else "(d)"
    ax.set_title(f"{panel_letter} {label} (Ch {ch})", fontweight="bold")
    ax.set_xlabel("Time sample")
    ax.set_ylabel("Amplitude")
    if row == 0:
        ax.legend(loc="upper right", fontsize=7)
    
    # (b/e) Zoomed detail
    ax = axes[row, 1]
    ax.plot(t_zoom, orig_zoom, color=C_ORIG, alpha=0.8, linewidth=0.9, label="Original")
    ax.plot(t_zoom, dp_zoom, color=C_DP, alpha=0.7, linewidth=0.7, label="DASPack")
    ax.plot(t_zoom, ad_zoom, color=C_AD, alpha=0.7, linewidth=0.7, label="Adaptive2D")
    panel_letter = "(b)" if row == 0 else "(e)"
    ax.set_title(f"{panel_letter} Zoomed detail", fontweight="bold")
    ax.set_xlabel("Time sample")
    ax.set_ylabel("Amplitude")
    if row == 0:
        ax.legend(loc="upper right", fontsize=7)
    
    # (c/f) Reconstruction error
    ax = axes[row, 2]
    ax.plot(t_zoom, err_dp_zoom, color=C_DP, alpha=0.7, linewidth=0.5, label="DASPack")
    ax.plot(t_zoom, err_ad_zoom, color=C_AD, alpha=0.7, linewidth=0.5, label="Adaptive2D")
    ax.axhline(epsilon, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axhline(-epsilon, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axhline(epsilon/2, color=C_AD, linestyle=":", alpha=0.3, linewidth=0.6)
    ax.axhline(-epsilon/2, color=C_AD, linestyle=":", alpha=0.3, linewidth=0.6)
    ax.fill_between(t_zoom, -epsilon, epsilon, color="green", alpha=0.04)
    panel_letter = "(c)" if row == 0 else "(f)"
    max_dp = np.max(np.abs(err_dp_zoom))
    max_ad = np.max(np.abs(err_ad_zoom))
    ax.set_title(f"{panel_letter} Reconstruction error", fontweight="bold")
    ax.set_xlabel("Time sample")
    ax.set_ylabel("Error")
    # Annotation
    if row == 0:
        ax.text(0.02, 0.95, f"max DASPack: {max_dp:.4f}\nmax Adaptive2D: {max_ad:.4f}",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc", alpha=0.9))
    ax.legend(loc="upper right", fontsize=7)

fig.suptitle(
    f"Waveform comparison $-$ Alboran dataset ($\\varepsilon$ = {epsilon:.3f})",
    fontsize=12, fontweight="bold", y=0.99
)
plt.tight_layout(pad=0.6, h_pad=0.8, w_pad=0.8)
fig.savefig(OUTPUT, bbox_inches="tight", facecolor="white", edgecolor="none")
plt.close()
print(f"\nSaved to {OUTPUT}")

# Print stats
for ch, label in [(HIGH_CH, "High"), (LOW_CH, "Low")]:
    orig = arr[ch, :]
    err_dp = np.abs(dp_restored[ch, :] - orig)
    err_ad = np.abs(ad_restored[ch, :] - orig)
    print(f"{label} (Ch {ch}): DASPack max_err={err_dp.max():.4f}, Adaptive2D max_err={err_ad.max():.4f}")
