"""
公平基准测试：在相同最大误差上界 ε 下对比所有压缩方法。

核心原则：
  - DASPack:       step = 2ε,  max error = ε
  - Adaptive2D:    高能量 step = ε (max error = ε/2), 低能量 step = 2ε (max error = ε)
  - ZFP:           accuracy = ε
  - Gzip:          quantize step = 2ε, max error = ε
  - JPEG2000:      quantize step = 2ε, max error = ε

  所有方法都保证 max |original - reconstructed| ≤ ε。

用法:
  cd /home/xdzhang/DAS_Compression/DASPack/improve/daspack-main/benchmark
  python fair_benchmark.py
"""

import os
import sys
import time
import numpy as np
import h5py
import struct
import pickle
import pandas as pd
from dataclasses import dataclass
from tempfile import NamedTemporaryFile

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from daspack import DASCoder, Quantizer
from adaptive_daspack_2d import AdaptiveDASPack2D


# ======================================================================
# 数据加载
# ======================================================================

def load_hdas_segment(h5_path):
    """从 HDAS .h5 文件加载 DAS 数据。"""
    with h5py.File(h5_path, "r") as f:
        data_raw = np.array(f["data"][:]).astype(np.float32)
        dt_s = float(f["data"].attrs["dt_s"])
        begin_iso = f["data"].attrs["begin_time"]
        dx_m = float(f["data"].attrs["dx_m"])
    return data_raw, dt_s, begin_iso, dx_m


# ======================================================================
# 压缩器封装（统一接口，以 ε 为误差容限）
# ======================================================================

@dataclass
class FairOptions:
    """以 ε 为误差容限的选项。所有方法保证 max error ≤ ε。"""
    epsilon: float  # 最大绝对误差容限


class DASPackFair:
    """DASPack，step = 2ε，max error = ε。"""
    name = "daspack"

    def __init__(self, threads=4):
        self.threads = threads
        self.coder = DASCoder(threads=threads)

    def write(self, arr, fn, opts):
        epsilon = opts.epsilon
        step = 2.0 * epsilon  # max error = step/2 = ε
        buf = self.coder.encode(arr.astype(np.float64), Quantizer.Uniform(step=step))
        with open(fn, "wb") as f:
            f.write(buf)

    def read(self, fn):
        with open(fn, "rb") as f:
            buf = f.read()
        return self.coder.decode(buf)


class ZfpFair:
    """ZFP，accuracy = ε。"""
    name = "zfp"

    def write(self, arr, fn, opts):
        import hdf5plugin
        epsilon = opts.epsilon
        with h5py.File(fn, "w") as f:
            dset = f.create_dataset(
                "data", data=arr,
                compression=hdf5plugin.Zfp(accuracy=float(epsilon)),
            )

    def read(self, fn):
        with h5py.File(fn) as f:
            return f["data"][:]


class GzipFair:
    """Gzip，先量化到 step=2ε 再 gzip。"""
    name = "gzip"

    def write(self, arr, fn, opts):
        epsilon = opts.epsilon
        step = 2.0 * epsilon
        data_int = np.round(arr.astype(np.float64) / step).astype(np.int32)
        with h5py.File(fn, "w") as f:
            dset = f.create_dataset("data", data=data_int, compression="gzip")
            dset.attrs["step"] = float(step)

    def read(self, fn):
        with h5py.File(fn) as f:
            dset = f["data"]
            step = float(dset.attrs["step"])
            data_int = dset[:]
        return data_int.astype(np.float64) * step


class JPEG2000Fair:
    """JPEG2000，先量化到 step=2ε 再编码。"""
    name = "jpeg2000"

    def write(self, arr, fn, opts):
        from imagecodecs import jpeg2k_encode
        epsilon = opts.epsilon
        step = 2.0 * epsilon
        data_int = np.round(arr.astype(np.float64) / step).astype(np.int32)
        encoded = jpeg2k_encode(data_int, reversible=True, numthreads=1)
        with h5py.File(fn, "w") as f:
            dset = f.create_dataset(
                "data", data=np.frombuffer(encoded, dtype=np.uint8)
            )
            dset.attrs["step"] = float(step)

    def read(self, fn):
        from imagecodecs import jpeg2k_decode
        with h5py.File(fn) as f:
            raw = f["data"][:].tobytes()
            step = float(f["data"].attrs["step"])
        decoded = jpeg2k_decode(raw)
        return decoded.astype(np.float64) * step


# ======================================================================
# 评估指标
# ======================================================================

def compute_metrics(original, restored, block_map=None):
    """计算压缩质量指标。"""
    original = original.astype(np.float64)
    restored = restored.astype(np.float64)
    diff = restored - original

    metrics = {
        "max_abs_err": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mse": float(np.mean(diff**2)) if diff.size else 0.0,
        "rmse": float(np.sqrt(np.mean(diff**2))) if diff.size else 0.0,
    }

    # SNR (dB)
    signal_power = float(np.mean(original**2))
    noise_power = float(np.mean(diff**2))
    metrics["snr_db"] = float(10 * np.log10(signal_power / noise_power)) if noise_power > 0 else float("inf")

    # 分区域统计
    if block_map is not None:
        high_mask = block_map == 1
        low_mask = block_map == 0
        if high_mask.any():
            metrics["mse_high"] = float(np.mean(diff[high_mask]**2))
            metrics["snr_high_db"] = float(
                10 * np.log10(np.mean(original[high_mask]**2) / np.mean(diff[high_mask]**2))
            ) if np.mean(diff[high_mask]**2) > 0 else float("inf")
        else:
            metrics["mse_high"] = 0.0
            metrics["snr_high_db"] = float("inf")

        if low_mask.any():
            metrics["mse_low"] = float(np.mean(diff[low_mask]**2))
            metrics["snr_low_db"] = float(
                10 * np.log10(np.mean(original[low_mask]**2) / np.mean(diff[low_mask]**2))
            ) if np.mean(diff[low_mask]**2) > 0 else float("inf")
        else:
            metrics["mse_low"] = 0.0
            metrics["snr_low_db"] = float("inf")

    return metrics


# ======================================================================
# 基准测试主函数
# ======================================================================

def run_fair_benchmark(datasets, epsilons, block_channels=500, block_samples=500,
                       threshold_factor=1.5, threads=4, output_csv="fair_benchmark_results.csv"):
    """
    在相同误差上界 ε 下运行公平基准测试。

    Parameters
    ----------
    datasets : dict
        {name: ndarray} 数据集字典。
    epsilons : list of float
        误差容限列表。
    block_channels, block_samples : int
        2D 块大小。
    threshold_factor : float
        能量阈值因子 τ。
    threads : int
        并行线程数。
    output_csv : str
        结果输出 CSV 文件名。
    """
    # 初始化压缩器
    codecs = [
        DASPackFair(threads=threads),
        AdaptiveDASPack2D(
            tolerance=0.1,  # 会被 opts 覆盖
            block_channels=block_channels,
            block_samples=block_samples,
            threshold_factor=threshold_factor,
            threads=threads,
        ),
        ZfpFair(),
        GzipFair(),
        JPEG2000Fair(),
    ]

    rows = []

    for ds_name, arr in datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}  shape={arr.shape}  dtype={arr.dtype}")
        print(f"{'='*60}")

        # 预计算块分类图（用于分区域统计）
        adaptive_temp = AdaptiveDASPack2D(
            tolerance=0.1,
            block_channels=block_channels,
            block_samples=block_samples,
            threshold_factor=threshold_factor,
        )
        block_map, block_info = adaptive_temp.get_block_map(arr)
        n_high = sum(1 for b in block_info if b["is_high"])
        n_low = sum(1 for b in block_info if not b["is_high"])
        print(f"  Blocks: {len(block_info)} total, {n_high} high-energy, {n_low} low-energy")

        for epsilon in epsilons:
            print(f"\n  ε = {epsilon}")
            opts = FairOptions(epsilon=epsilon)
            original_bytes = int(arr.nbytes)

            for codec in codecs:
                codec_name = codec.name
                print(f"    {codec_name} ...", end=" ", flush=True)

                tmp = NamedTemporaryFile(suffix=".h5", delete=False)
                tmp_name = tmp.name
                tmp.close()

                try:
                    # 编码
                    t0 = time.perf_counter()
                    codec.write(arr, tmp_name, opts)
                    enc_time = time.perf_counter() - t0

                    compressed_bytes = os.path.getsize(tmp_name)

                    # 解码
                    t0 = time.perf_counter()
                    restored = codec.read(tmp_name)
                    dec_time = time.perf_counter() - t0

                    # 指标
                    metrics = compute_metrics(arr, restored, block_map)
                    verified = metrics["max_abs_err"] <= epsilon + 1e-6

                    cr = original_bytes / compressed_bytes if compressed_bytes > 0 else float("inf")

                    row = {
                        "dataset": ds_name,
                        "codec": codec_name,
                        "epsilon": epsilon,
                        "daspack_step": 2.0 * epsilon,
                        "compression_ratio": cr,
                        "original_bytes": original_bytes,
                        "compressed_bytes": compressed_bytes,
                        "enc_time_s": enc_time,
                        "dec_time_s": dec_time,
                        "verified": verified,
                        **metrics,
                    }
                    rows.append(row)

                    status = "✓" if verified else "✗"
                    print(
                        f"CR={cr:.2f}  MaxAE={metrics['max_abs_err']:.6f}  "
                        f"MSE={metrics['mse']:.2e}  SNR={metrics['snr_db']:.1f}dB  "
                        f"enc={enc_time:.2f}s  [{status}]"
                    )

                except Exception as e:
                    print(f"ERROR: {e}")
                    rows.append({
                        "dataset": ds_name,
                        "codec": codec_name,
                        "epsilon": epsilon,
                        "error": str(e),
                    })
                finally:
                    try:
                        os.remove(tmp_name)
                    except OSError:
                        pass

    # 保存结果
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\n结果已保存到 {output_csv}")
    return df


# ======================================================================
# 主入口
# ======================================================================

if __name__ == "__main__":
    # 数据集路径
    data_dir = "./data"

    datasets = {
        "alboran": load_hdas_segment(os.path.join(data_dir, "7M_S_20231005-214004_RAW.h5"))[0],
        "castor": load_hdas_segment(os.path.join(data_dir, "ZI_C_20230609-102630_RAW.h5"))[0],
    }

    # 误差容限列表
    epsilons = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

    # 运行基准测试
    df = run_fair_benchmark(
        datasets=datasets,
        epsilons=epsilons,
        block_channels=500,
        block_samples=500,
        threshold_factor=1.5,
        threads=4,
        output_csv="fair_benchmark_results.csv",
    )

    # 打印汇总表
    print("\n" + "=" * 80)
    print("汇总：压缩率对比（相同 max error ≤ ε）")
    print("=" * 80)
    pivot = df.pivot_table(
        index=["dataset", "epsilon"],
        columns="codec",
        values="compression_ratio",
        aggfunc="first",
    )
    print(pivot.to_string(float_format="%.2f"))

    print("\n" + "=" * 80)
    print("汇总：MSE 对比")
    print("=" * 80)
    pivot_mse = df.pivot_table(
        index=["dataset", "epsilon"],
        columns="codec",
        values="mse",
        aggfunc="first",
    )
    print(pivot_mse.to_string(float_format="%.2e"))

    if "mse_high" in df.columns:
        print("\n" + "=" * 80)
        print("汇总：高能量区域 MSE 对比")
        print("=" * 80)
        pivot_mse_high = df.pivot_table(
            index=["dataset", "epsilon"],
            columns="codec",
            values="mse_high",
            aggfunc="first",
        )
        print(pivot_mse_high.to_string(float_format="%.2e"))
