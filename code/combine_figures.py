"""
重新生成紧凑版图表：减小高度、降低 DPI、收紧边距。
目标：每张图高度减少 30-40%，使图文混排更紧凑。
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import shutil

BASE = "/home/xdzhang/DAS_Compression/DASPack/improve/daspack-main/benchmark"
OUT = "/home/xdzhang/DAS_Compression/DASPack/figures"

def load_img(path):
    return mpimg.imread(path)

def add_img(ax, path, title=""):
    img = load_img(path)
    ax.imshow(img)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold", pad=3)

# ================================================================
# Figure 1: 率失真曲线 — 上下排列，减小高度
# ================================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 6))
add_img(axes[0], os.path.join(BASE, "rate_distortion_alboran.png"), "(a) Alboran")
add_img(axes[1], os.path.join(BASE, "rate_distortion_castor.png"), "(b) Castor")
plt.tight_layout(pad=0.2, h_pad=0.3)
fig.savefig(os.path.join(OUT, "fig1_rate_distortion.png"), dpi=120, bbox_inches="tight")
plt.close()
print("Figure 1 saved (compact)")

# ================================================================
# Figure 2: 波形对比 — 高能量+低能量 左右排列，降低高度
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
add_img(axes[0], os.path.join(BASE, "waveform_plots/waveform_alboran_high.png"), "(a) High-energy (ch 250)")
add_img(axes[1], os.path.join(BASE, "waveform_plots/waveform_alboran_low.png"), "(b) Low-energy (ch 750)")
plt.tight_layout(pad=0.2, w_pad=0.3)
fig.savefig(os.path.join(OUT, "fig2_waveform.png"), dpi=120, bbox_inches="tight")
plt.close()
print("Figure 2 saved (compact, side-by-side)")

# ================================================================
# Figure 3: 误差地图 + 瀑布图 — 左右排列，大幅降低高度
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
add_img(axes[0], os.path.join(BASE, "waveform_plots/errormap_alboran.png"), "(a) Error map")
add_img(axes[1], os.path.join(BASE, "waveform_plots/waterfall_alboran.png"), "(b) Waterfall")
plt.tight_layout(pad=0.2, w_pad=0.3)
fig.savefig(os.path.join(OUT, "fig3_error_map.png"), dpi=120, bbox_inches="tight")
plt.close()
print("Figure 3 saved (compact, side-by-side)")

# ================================================================
# Figure 4: 参数敏感性 — 左右排列，减半高度
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
add_img(axes[0], os.path.join(BASE, "param_sensitivity_plots/sweep_block_size.png"), "(a) Block size")
add_img(axes[1], os.path.join(BASE, "param_sensitivity_plots/sweep_fraction_normalized.png"), "(b) Fraction (normalized)")
plt.tight_layout(pad=0.2, w_pad=0.3)
fig.savefig(os.path.join(OUT, "fig4_sensitivity.png"), dpi=120, bbox_inches="tight")
plt.close()
print("Figure 4 saved (compact, side-by-side)")

# ================================================================
# Figure 5: 汇总柱状图 (保留，但降低 DPI)
# ================================================================
# 读取原图并重新保存为更低 DPI
img = mpimg.imread(os.path.join(BASE, "waveform_plots/summary_all_datasets.png"))
fig, ax = plt.subplots(figsize=(12, 5))
ax.imshow(img)
ax.axis("off")
plt.tight_layout(pad=0.1)
fig.savefig(os.path.join(OUT, "fig5_summary.png"), dpi=120, bbox_inches="tight")
plt.close()
print("Figure 5 saved (compact)")

print(f"\nDone! 5 compact figures saved to {OUT}/")
