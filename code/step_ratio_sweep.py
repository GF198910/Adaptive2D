"""
步长比 ρ 扫描实验：验证论文默认的 2:1 配置是否接近最优。

参数化（保持 max|error| ≤ ε 的认证界对所有 ρ 成立）：
  - 低能量块步长固定为 2ε（界的上限）
  - 高能量块步长 = 2ε/ρ，ρ ∈ {1.0, 1.5, 2.0, 3.0, 4.0}
  - ρ = 1 退化为均匀量化（≈ DASPack 基线）；ρ = 2 为论文默认配置

实现：AdaptiveDASPack2D(tolerance=2ε/ρ, step_ratio=ρ)，此时
  high step = 2ε/ρ，low step = ρ × (2ε/ρ) = 2ε。

注意：噪声阈值化在本实验中关闭。因为内部容限为 2ε/ρ，
当 ρ < 2 时置零阈值 2ε/ρ > ε 会破坏 ε 界；为保证所有 ρ
配置严格满足同一界，统一关闭（对所有 ρ 一致，相对比较公平）。

指标：CR、overall MSE、high-energy MSE（按 fraction=0.2 的块分类）、
low-energy MSE、实测 max|error|（验证 ≤ ε）。

用法:
  /home/xdzhang/anaconda3/bin/python /home/xdzhang/DAS_Compression/GJI/code/step_ratio_sweep.py
"""

import os
import sys
import time
import numpy as np
import h5py
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_DASPACK_SRC = os.path.join(_HERE, "daspack_src")
_DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "data"))
_RESULTS_DIR = os.path.normpath(os.path.join(_HERE, "..", "results"))

sys.path.insert(0, _DASPACK_SRC)
sys.path.insert(0, _HERE)

from adaptive_daspack_2d import AdaptiveDASPack2D  # noqa: E402

DATASETS = {
    "alboran": "7M_S_20231005-214004_RAW.h5",
    "castor": "ZI_C_20230609-102630_RAW.h5",
    "athens": "athens.h5",
}

RATIOS = [1.0, 1.5, 2.0, 3.0, 4.0]
REL_TOLS = [0.001, 0.002, 0.005]
FRACTION = 0.2
BLOCK_CH, BLOCK_SP = 250, 500


def load_dataset(name):
    path = os.path.join(_DATA_DIR, DATASETS[name])
    with h5py.File(path, "r") as f:
        return np.array(f["data"][:]).astype(np.float64)


def high_energy_mask(data):
    """按论文默认分类（percentile, fraction=0.2）计算高能量块掩码。"""
    codec = AdaptiveDASPack2D(
        tolerance=1.0, block_channels=BLOCK_CH, block_samples=BLOCK_SP,
        high_energy_fraction=FRACTION, noise_thresholding=False)
    blocks = codec._compute_block_energies(data)
    is_high, _ = codec._determine_threshold(blocks, data)
    mask = np.zeros(data.shape, dtype=bool)
    for info, h in zip(blocks, is_high):
        (c0, c1), (t0, t1) = info["ch_range"], info["t_range"]
        mask[c0:c1, t0:t1] = h
    return mask


def run_one(data, epsilon, rho, threads=4):
    """以参数化 (2ε/ρ, ρ) 压缩-解压，返回指标 dict。"""
    codec = AdaptiveDASPack2D(
        tolerance=2.0 * epsilon / rho,
        block_channels=BLOCK_CH,
        block_samples=BLOCK_SP,
        threshold_mode="percentile",
        high_energy_fraction=FRACTION,
        noise_thresholding=False,   # 见模块 docstring
        step_ratio=rho,
        threads=threads,
    )
    t0 = time.perf_counter()
    buf = codec.compress(data)
    comp_t = time.perf_counter() - t0
    restored = codec.decompress(buf)

    # 压缩率：原始 float64 字节数 / 完整码流字节数（含元数据）
    cr = data.nbytes / len(buf)

    err = restored - data
    max_err = float(np.max(np.abs(err)))
    return restored, {
        "cr": cr,
        "max_err": max_err,
        "compress_time_s": comp_t,
    }


def main():
    threads = 4
    rows = []

    for ds_name in DATASETS:
        print(f"\n{'='*80}\n数据集: {ds_name}\n{'='*80}")
        data = load_dataset(ds_name)
        data_std = float(np.std(data))
        mask = high_energy_mask(data)
        print(f"  shape={data.shape}  std={data_std:.4f}  "
              f"high-energy 样本占比={mask.mean()*100:.1f}%")

        for rel_tol in REL_TOLS:
            epsilon = rel_tol * data_std
            for rho in RATIOS:
                restored, m = run_one(data, epsilon, rho, threads)
                err = restored - data
                mse_all = float(np.mean(err ** 2))
                mse_hi = float(np.mean(err[mask] ** 2))
                mse_lo = float(np.mean(err[~mask] ** 2))
                ok = "OK" if m["max_err"] <= epsilon * 1.0001 else "VIOLATION"
                print(f"  r={rel_tol:<7} ρ={rho:<4} CR={m['cr']:.2f}  "
                      f"MSE_hi={mse_hi:.4e}  MSE_lo={mse_lo:.4e}  "
                      f"max|err|={m['max_err']:.4f} (ε={epsilon:.4f}) {ok}")
                rows.append({
                    "dataset": ds_name, "rel_tol": rel_tol,
                    "epsilon": epsilon, "rho": rho,
                    "cr": m["cr"], "mse_overall": mse_all,
                    "mse_high": mse_hi, "mse_low": mse_lo,
                    "max_err": m["max_err"],
                    "bound_ok": m["max_err"] <= epsilon * 1.0001,
                    "compress_time_s": m["compress_time_s"],
                })

    os.makedirs(_RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    out = os.path.join(_RESULTS_DIR, "step_ratio_sweep.csv")
    df.to_csv(out, index=False)
    print(f"\n结果已保存到 {out}")

    # 汇总：相对 ρ=1（均匀量化基线）的归一化指标
    print("\n" + "=" * 90)
    print("汇总：相对 ρ=1（均匀基线）的 high-energy MSE 比与 CR 损失")
    print("=" * 90)
    print(f"{'dataset':>10} {'rel_tol':>8} {'ρ':>4} "
          f"{'CR':>7} {'CR loss%':>9} {'MSE_hi/ρ1':>10} {'theory 1/ρ²':>11}")
    print("-" * 90)
    for ds_name in DATASETS:
        for rel_tol in REL_TOLS:
            sub = df[(df.dataset == ds_name) & (df.rel_tol == rel_tol)]
            base = sub[sub.rho == 1.0].iloc[0]
            for _, r in sub.iterrows():
                print(f"{ds_name:>10} {rel_tol:>8.3f} {r['rho']:>4.1f} "
                      f"{r['cr']:>7.2f} "
                      f"{(1 - r['cr']/base['cr'])*100:>8.1f}% "
                      f"{r['mse_high']/base['mse_high']:>10.3f} "
                      f"{1/r['rho']**2:>11.3f}")


if __name__ == "__main__":
    main()
