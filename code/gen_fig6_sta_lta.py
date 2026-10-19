"""
生成 STA/LTA 检测性能图（Figure 6）
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "grid.linestyle": "--",
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#cccccc",
})

C_DP = "#2166ac"
C_AD = "#b2182b"

BASE = "/home/xdzhang/DAS_Compression/DASPack/improve/daspack-main/benchmark"
OUT = "/home/xdzhang/DAS_Compression/DASPack/figures"

df = pd.read_csv(os.path.join(BASE, "sta_lta_multi_tolerance.csv"))

# 过滤出 daspack 和 adaptive
dp = df[df["method"] == "daspack"].sort_values("rel_tol")
ad = df[df["method"] == "adaptive_2d_thresh"].sort_values("rel_tol")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

# (a) F1 score vs rel_tol
ax = axes[0]
ax.plot(dp["rel_tol"] * 1000, dp["f1"], "o-", color=C_DP, label="DASPack",
        markeredgecolor="white", markeredgewidth=0.5)
ax.plot(ad["rel_tol"] * 1000, ad["f1"], "s-", color=C_AD, label="Adaptive2D",
        markeredgecolor="white", markeredgewidth=0.5)
ax.set_xlabel("Relative tolerance ($r \\times 10^3$)")
ax.set_ylabel("F1 Score")
ax.set_title("(a) Detection F1 Score vs. Error Tolerance", fontweight="bold")
ax.legend(loc="lower left")
ax.set_xscale("log")
ax.set_ylim(0.98, 1.002)

# (b) Mean |Δt| vs rel_tol
ax = axes[1]
ax.plot(dp["rel_tol"] * 1000, dp["mean_abs_time_diff"], "o-", color=C_DP, label="DASPack",
        markeredgecolor="white", markeredgewidth=0.5)
ax.plot(ad["rel_tol"] * 1000, ad["mean_abs_time_diff"], "s-", color=C_AD, label="Adaptive2D",
        markeredgecolor="white", markeredgewidth=0.5)
ax.set_xlabel("Relative tolerance ($r \\times 10^3$)")
ax.set_ylabel("Mean $|\\Delta t|$ (samples)")
ax.set_title("(b) Event Timing Deviation vs. Error Tolerance", fontweight="bold")
ax.legend(loc="upper left")
ax.set_xscale("log")

plt.tight_layout(pad=0.5, w_pad=1.0)
fig.savefig(os.path.join(OUT, "fig6_sta_lta.png"), bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print("Figure 6 saved (STA/LTA detection performance)")
