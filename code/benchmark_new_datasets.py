"""
对新数据集 9G_P 和 athens 运行基准测试。

在 rel_tol=0.001 下测试 DASPack 和 Adaptive2D，记录压缩率、误差、MSE、SNR 等指标。
ε = rel_tol × std(data)
  - DASPack:      step = 2ε, max error = ε
  - Adaptive2D:   高能量 step = ε (max error = ε/2), 低能量 step = 2ε (max error = ε)

用法:
  cd /home/xdzhang/DAS_Compression/DASPack/improve/daspack-main/benchmark
  python benchmark_new_datasets.py
"""

import os
import sys
import time
import numpy as np
import h5py
import pandas as pd
from tempfile import NamedTemporaryFile

# 添加路径：当前目录（用于 adaptive_daspack_2d, fair_benchmark）和父目录（用于 daspack）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from daspack import DASCoder, Quantizer
from adaptive_daspack_2d import AdaptiveDASPack2D
from fair_benchmark import DASPackFair, FairOptions, compute_metrics


# ======================================================================
# 数据加载
# ======================================================================

DATA_DIR = "./data"

def load_dataset(name):
    """从 HDF5 文件加载 DAS 数据（'data' 键）。"""
    path = os.path.join(DATA_DIR, f"{name}.h5")
    with h5py.File(path, "r") as f:
        data = np.array(f["data"][:]).astype(np.float64)
    return data


# ======================================================================
# 基准测试
# ======================================================================

def run_benchmark(datasets, rel_tol=0.001, block_channels=250,
                  block_samples=500, high_energy_fraction=0.2,
                  threads=4, output_csv="new_datasets_results.csv"):
    """
    对新数据集运行 DASPack 和 Adaptive2D 基准测试。

    Parameters
    ----------
    datasets : dict
        {name: ndarray} 数据集字典。
    rel_tol : float
        相对误差容限。ε = rel_tol × std(data)。
    block_channels, block_samples : int
        2D 块大小。
    high_energy_fraction : float
        高能量块比例（percentile 模式）。
    threads : int
        并行线程数。
    output_csv : str
        结果输出 CSV 文件名。
    """
    rows = []

    for ds_name, arr in datasets.items():
        data_std = float(np.std(arr))
        epsilon = rel_tol * data_std
        channels, samples = arr.shape
        original_bytes = channels * samples * 8  # float64

        print(f"\n{'='*70}")
        print(f"Dataset: {ds_name}  shape={arr.shape}  std={data_std:.4f}")
        print(f"  rel_tol={rel_tol}  ε={epsilon:.6f}")
        print(f"  original_bytes={original_bytes} ({channels}×{samples}×8)")
        print(f"{'='*70}")

        # 预计算块分类图（用于分区域 MSE/SNR 统计）
        adaptive_temp = AdaptiveDASPack2D(
            tolerance=epsilon,
            block_channels=min(block_channels, channels),
            block_samples=block_samples,
            threshold_mode="percentile",
            high_energy_fraction=high_energy_fraction,
        )
        block_map, block_info = adaptive_temp.get_block_map(arr)
        n_high = sum(1 for b in block_info if b["is_high"])
        n_low = sum(1 for b in block_info if not b["is_high"])
        print(f"  Blocks: {len(block_info)} (high={n_high}, low={n_low})")

        # 定义 codecs
        codecs_factory = {
            "daspack": lambda: DASPackFair(threads=threads),
            "adaptive_2d": lambda: AdaptiveDASPack2D(
                tolerance=epsilon,
                block_channels=block_channels,
                block_samples=block_samples,
                threshold_mode="percentile",
                high_energy_fraction=high_energy_fraction,
                threads=threads,
            ),
        }

        opts = FairOptions(epsilon=epsilon)

        for codec_name, codec_factory in codecs_factory.items():
            codec = codec_factory()
            print(f"  {codec_name} ...", end=" ", flush=True)

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
                    "max_abs_err": metrics["max_abs_err"],
                    "mse": metrics["mse"],
                    "mse_high": metrics.get("mse_high", 0.0),
                    "mse_low": metrics.get("mse_low", 0.0),
                    "snr_db": metrics["snr_db"],
                    "snr_high_db": metrics.get("snr_high_db", float("inf")),
                    "n_high_blocks": n_high,
                    "n_low_blocks": n_low,
                    "verified": verified,
                }
                rows.append(row)

                status = "✓" if verified else "✗"
                print(
                    f"CR={cr:.2f}  MaxAE={metrics['max_abs_err']:.6f}  "
                    f"MSE={metrics['mse']:.2e}  "
                    f"MSE_high={metrics.get('mse_high', 0):.2e}  "
                    f"MSE_low={metrics.get('mse_low', 0):.2e}  "
                    f"enc={enc_time:.2f}s  [{status}]"
                )

            except Exception as e:
                print(f"ERROR: {e}")
                rows.append({
                    "dataset": ds_name,
                    "codec": codec_name,
                    "rel_tol": rel_tol,
                    "epsilon": epsilon,
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
# 主入口
# ======================================================================

if __name__ == "__main__":
    print("加载新数据集...")
    datasets = {}
    for name in ["9G_P", "athens"]:
        datasets[name] = load_dataset(name)
        print(f"  {name}: {datasets[name].shape}  std={np.std(datasets[name]):.4f}")

    # 运行基准测试
    df = run_benchmark(
        datasets=datasets,
        rel_tol=0.001,
        block_channels=250,
        block_samples=500,
        high_energy_fraction=0.2,
        threads=4,
        output_csv="new_datasets_results.csv",
    )

    # ==================================================================
    # 汇总
    # ==================================================================
    print("\n" + "=" * 80)
    print("汇总：压缩率对比（rel_tol=0.001）")
    print("=" * 80)
    pivot_cr = df.pivot_table(
        index="dataset", columns="codec",
        values="compression_ratio", aggfunc="first"
    )
    print(pivot_cr.to_string(float_format="%.2f"))

    print("\n" + "=" * 80)
    print("汇总：最大绝对误差对比")
    print("=" * 80)
    pivot_mae = df.pivot_table(
        index="dataset", columns="codec",
        values="max_abs_err", aggfunc="first"
    )
    print(pivot_mae.to_string(float_format="%.6f"))

    print("\n" + "=" * 80)
    print("汇总：MSE 对比")
    print("=" * 80)
    pivot_mse = df.pivot_table(
        index="dataset", columns="codec",
        values="mse", aggfunc="first"
    )
    print(pivot_mse.to_string(float_format="%.2e"))

    print("\n" + "=" * 80)
    print("汇总：高能量区域 MSE 对比")
    print("=" * 80)
    pivot_mse_high = df.pivot_table(
        index="dataset", columns="codec",
        values="mse_high", aggfunc="first"
    )
    print(pivot_mse_high.to_string(float_format="%.2e"))

    print("\n" + "=" * 80)
    print("汇总：低能量区域 MSE 对比")
    print("=" * 80)
    pivot_mse_low = df.pivot_table(
        index="dataset", columns="codec",
        values="mse_low", aggfunc="first"
    )
    print(pivot_mse_low.to_string(float_format="%.2e"))

    print("\n" + "=" * 80)
    print("汇总：SNR 对比 (dB)")
    print("=" * 80)
    pivot_snr = df.pivot_table(
        index="dataset", columns="codec",
        values="snr_db", aggfunc="first"
    )
    print(pivot_snr.to_string(float_format="%.2f"))

    print("\n" + "=" * 80)
    print("汇总：高能量区域 SNR 对比 (dB)")
    print("=" * 80)
    pivot_snr_high = df.pivot_table(
        index="dataset", columns="codec",
        values="snr_high_db", aggfunc="first"
    )
    print(pivot_snr_high.to_string(float_format="%.2f"))

    print("\n" + "=" * 80)
    print("验证结果（max_abs_err ≤ ε）")
    print("=" * 80)
    for _, row in df.iterrows():
        status = "✓ PASS" if row["verified"] else "✗ FAIL"
        print(f"  {row['dataset']:10s} / {row['codec']:12s}  "
              f"max_abs_err={row['max_abs_err']:.6f}  ε={row['epsilon']:.6f}  [{status}]")

    print("\n" + "=" * 80)
    print("块统计")
    print("=" * 80)
    for ds_name in df["dataset"].unique():
        sub = df[df["dataset"] == ds_name].iloc[0]
        print(f"  {ds_name:10s}: n_high_blocks={sub['n_high_blocks']}, n_low_blocks={sub['n_low_blocks']}")
