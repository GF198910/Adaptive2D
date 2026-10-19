"""
综合对比基准测试：DASPack vs Adaptive2D vs ZFP vs JPEG2000

在 rel_tol=0.001（ε = rel_tol × std）下，对 7 个 DAS 数据集
对比四种压缩方法的压缩率、误差和速度。

方法说明:
  - DASPack:     step = 2ε,  max error = ε
  - Adaptive2D:  高能量块 step=ε (max err=ε/2), 低能量块 step=2ε (max err=ε)
  - ZFP:         tolerance=ε (fixed-accuracy 模式)
  - JPEG2000:    归一化到 [0,255] uint8 后用 Pillow 编码 (不保证 bounded error)

所有方法的高/低能量区域 MSE 使用 Adaptive2D 的块分类图统一划分（公平对比）。

用法:
  cd /home/xdzhang/DAS_Compression/DASPack/improve/daspack-main/benchmark
  python comprehensive_benchmark.py
"""

import os
import sys
import time
import struct
import pickle
import io
import traceback

import numpy as np
import h5py
import pandas as pd

# ======================================================================
# sys.path 设置：daspack 包在父目录
# ======================================================================

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BENCH_DIR)
sys.path.insert(0, BENCH_DIR)
sys.path.insert(0, PARENT_DIR)

from daspack import DASCoder, Quantizer
from adaptive_daspack_2d import AdaptiveDASPack2D
from fair_benchmark import compute_metrics, FairOptions


# ======================================================================
# 配置
# ======================================================================

DATA_DIR = os.path.join(BENCH_DIR, "data")
REL_TOL = 0.001          # ε = rel_tol × std(data)
THREADS = 4
BLOCK_CHANNELS = 250
BLOCK_SAMPLES = 500
HIGH_ENERGY_FRACTION = 0.2
JPEG2000_QUALITY = 0.1   # bits per pixel (rates 模式)
OUTPUT_CSV = "comprehensive_benchmark_results.csv"

DATASET_FILES = {
    "alboran":       "7M_S_20231005-214004_RAW.h5",
    "castor":        "ZI_C_20230609-102630_RAW.h5",
    "event_0322_a":  "event_0322_a.h5",
    "event_0322_b":  "event_0322_b.h5",
    "event_0620_a":  "event_0620_a.h5",
    "event_0620_b":  "event_0620_b.h5",
    "event_0620_c":  "event_0620_c.h5",
}


# ======================================================================
# 数据加载
# ======================================================================

def load_all_datasets():
    """加载全部 7 个数据集，返回 {name: float32 ndarray}。"""
    datasets = {}
    for name, fn in DATASET_FILES.items():
        path = os.path.join(DATA_DIR, fn)
        if not os.path.exists(path):
            print(f"  [跳过] {name}: 文件不存在 {path}")
            continue
        with h5py.File(path, "r") as f:
            datasets[name] = np.array(f["data"][:]).astype(np.float32)
    return datasets


# ======================================================================
# 压缩方法实现（统一返回: compressed_bytes, restored, enc_time, dec_time）
# ======================================================================

def run_daspack(arr, epsilon):
    """DASPack: step = 2ε, max error = ε。"""
    coder = DASCoder(threads=THREADS)
    step = 2.0 * epsilon
    data = arr.astype(np.float64)

    t0 = time.perf_counter()
    compressed = coder.encode(data, Quantizer.Uniform(step=step))
    enc_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    restored = coder.decode(compressed)
    dec_time = time.perf_counter() - t0

    return compressed, restored, enc_time, dec_time


def run_adaptive2d(arr, epsilon):
    """Adaptive2D: 高能量 step=ε, 低能量 step=2ε + 噪声阈值化。"""
    codec = AdaptiveDASPack2D(
        tolerance=epsilon,
        block_channels=BLOCK_CHANNELS,
        block_samples=BLOCK_SAMPLES,
        threshold_mode="percentile",
        high_energy_fraction=HIGH_ENERGY_FRACTION,
        threads=THREADS,
    )
    opts = FairOptions(epsilon=epsilon)

    t0 = time.perf_counter()
    compressed = codec.compress(arr, opts)
    enc_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    restored = codec.decompress(compressed)
    dec_time = time.perf_counter() - t0

    return compressed, restored, enc_time, dec_time


def run_zfp(arr, epsilon):
    """ZFP: fixed-accuracy 模式, tolerance=ε。"""
    import zfpy
    data = arr.astype(np.float64)

    t0 = time.perf_counter()
    compressed = zfpy.compress_numpy(data, tolerance=float(epsilon))
    enc_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    restored = zfpy.decompress_numpy(compressed)
    dec_time = time.perf_counter() - t0

    return compressed, restored, enc_time, dec_time


def _jpeg2000_pillow(scaled):
    """用 Pillow 编码 JPEG2000，返回 bytes。"""
    from PIL import Image
    img = Image.fromarray(scaled, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="JPEG2000", irreversible=True,
             quality_mode="rates", quality=JPEG2000_QUALITY)
    return buf.getvalue()


def _jpeg2000_imagecodecs(scaled):
    """用 imagecodecs 编码 JPEG2000（Pillow 失败时的后备）。"""
    from imagecodecs import jpeg2k_encode
    encoded = jpeg2k_encode(scaled, reversible=False, numthreads=1)
    return bytes(encoded)


def run_jpeg2000(arr, epsilon):
    """JPEG2000: 归一化到 [0,255] uint8 后压缩（不保证 bounded error）。"""
    data = arr.astype(np.float64)

    t0 = time.perf_counter()
    arr_min = float(data.min())
    arr_max = float(data.max())
    rng = arr_max - arr_min
    if rng == 0:
        rng = 1.0
    scaled = ((data - arr_min) / rng * 255.0).astype(np.uint8)

    # 编码 JPEG2000（Pillow 优先，imagecodecs 后备）
    try:
        jpeg_bytes = _jpeg2000_pillow(scaled)
    except Exception:
        try:
            import glymur  # noqa: F401
            # glymur 基于 OpenJPEG，API 为文件式，此处用 imagecodecs 替代
            jpeg_bytes = _jpeg2000_imagecodecs(scaled)
        except Exception:
            jpeg_bytes = _jpeg2000_imagecodecs(scaled)

    # 打包元数据（arr_min, arr_max, shape）+ JPEG2000 比特流
    meta = pickle.dumps({
        "arr_min": arr_min, "arr_max": arr_max, "shape": arr.shape,
    })
    compressed = struct.pack("<I", len(meta)) + meta + jpeg_bytes
    enc_time = time.perf_counter() - t0

    # 解码
    t0 = time.perf_counter()
    pos = 0
    meta_len = struct.unpack_from("<I", compressed, pos)[0]
    pos += 4
    meta_d = pickle.loads(compressed[pos:pos + meta_len])
    pos += meta_len
    jpeg_stream = compressed[pos:]

    try:
        from PIL import Image
        buf = io.BytesIO(jpeg_stream)
        img = Image.open(buf)
        scaled2 = np.array(img, dtype=np.uint8)
    except Exception:
        from imagecodecs import jpeg2k_decode
        scaled2 = np.asarray(jpeg2k_decode(jpeg_stream), dtype=np.uint8)

    rng_d = meta_d["arr_max"] - meta_d["arr_min"]
    if rng_d == 0:
        rng_d = 1.0
    restored = scaled2.astype(np.float64) / 255.0 * rng_d + meta_d["arr_min"]
    dec_time = time.perf_counter() - t0

    return compressed, restored, enc_time, dec_time


# ======================================================================
# 方法注册表
# ======================================================================

METHODS = [
    ("DASPack",    run_daspack),
    ("Adaptive2D", run_adaptive2d),
    ("ZFP",        run_zfp),
    ("JPEG2000",   run_jpeg2000),
]


# ======================================================================
# 主基准测试流程
# ======================================================================

def main():
    # 禁用 PIL 的 DecompressionBomb 警告
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
    except Exception:
        pass

    print("=" * 80)
    print("综合对比基准测试：DASPack vs Adaptive2D vs ZFP vs JPEG2000")
    print(f"rel_tol = {REL_TOL}  (ε = {REL_TOL} × std)")
    print("=" * 80)

    # 加载数据
    print("\n加载数据集...")
    datasets = load_all_datasets()
    print(f"共 {len(datasets)} 个数据集:")
    for name, arr in datasets.items():
        print(f"  {name:16s}  shape={str(arr.shape):16s}  std={np.std(arr):.4f}")

    rows = []

    for ds_name, arr in datasets.items():
        data_std = float(np.std(arr))
        epsilon = REL_TOL * data_std
        channels, samples = arr.shape
        original_bytes = channels * samples * 8  # float64

        print(f"\n{'=' * 80}")
        print(f"Dataset: {ds_name}  shape={arr.shape}  std={data_std:.4f}  ε={epsilon:.6f}")
        print(f"{'=' * 80}")

        # 预计算块分类图（用于所有方法的 high/low MSE 划分）
        classifier = AdaptiveDASPack2D(
            tolerance=0.1,  # 不影响分类，仅影响 step 字段
            block_channels=BLOCK_CHANNELS,
            block_samples=BLOCK_SAMPLES,
            threshold_mode="percentile",
            high_energy_fraction=HIGH_ENERGY_FRACTION,
        )
        block_map, block_info = classifier.get_block_map(arr)
        n_high = sum(1 for b in block_info if b["is_high"])
        n_low = sum(1 for b in block_info if not b["is_high"])
        print(f"  Blocks: {len(block_info)} (high={n_high}, low={n_low})")

        for method_name, method_fn in METHODS:
            print(f"  {method_name:14s} ...", end=" ", flush=True)
            try:
                compressed, restored, enc_time, dec_time = method_fn(arr, epsilon)
                compressed_bytes = len(compressed)

                # 确保还原数据形状一致
                restored = np.asarray(restored).astype(np.float64)
                if restored.shape != arr.shape:
                    restored = restored.reshape(arr.shape)

                metrics = compute_metrics(arr, restored, block_map)
                cr = original_bytes / compressed_bytes if compressed_bytes > 0 else float("inf")
                verified = metrics["max_abs_err"] <= epsilon + 1e-6

                row = {
                    "dataset": ds_name,
                    "method": method_name,
                    "rel_tol": REL_TOL,
                    "epsilon": epsilon,
                    "data_std": data_std,
                    "channels": channels,
                    "samples": samples,
                    "original_bytes": original_bytes,
                    "compressed_bytes": compressed_bytes,
                    "compression_ratio": cr,
                    "max_abs_err": metrics["max_abs_err"],
                    "mse": metrics["mse"],
                    "mse_high": metrics.get("mse_high", 0.0),
                    "mse_low": metrics.get("mse_low", 0.0),
                    "snr_db": metrics["snr_db"],
                    "enc_time": enc_time,
                    "dec_time": dec_time,
                    "verified": verified,
                    "n_blocks": len(block_info),
                    "n_high_blocks": n_high,
                    "n_low_blocks": n_low,
                }
                rows.append(row)

                status = "✓" if verified else "✗"
                print(
                    f"CR={cr:7.2f}  MaxAE={metrics['max_abs_err']:.6f}  "
                    f"MSE={metrics['mse']:.2e}  "
                    f"MSE_h={metrics.get('mse_high', 0):.2e}  "
                    f"MSE_l={metrics.get('mse_low', 0):.2e}  "
                    f"SNR={metrics['snr_db']:6.1f}dB  "
                    f"enc={enc_time:.2f}s  dec={dec_time:.2f}s  [{status}]"
                )

            except Exception as e:
                print(f"ERROR: {e}")
                traceback.print_exc()
                rows.append({
                    "dataset": ds_name,
                    "method": method_name,
                    "rel_tol": REL_TOL,
                    "epsilon": epsilon,
                    "error": str(e),
                })

    # ==================================================================
    # 保存结果
    # ==================================================================

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n结果已保存到 {OUTPUT_CSV}")

    # ==================================================================
    # 汇总对比表
    # ==================================================================

    # 只取成功完成的行
    df_ok = df[df.get("error", pd.Series([False] * len(df))).isna()] if "error" in df.columns else df.copy()

    print("\n" + "=" * 80)
    print("汇总 1：压缩率 (Compression Ratio)")
    print("=" * 80)
    try:
        pivot_cr = df_ok.pivot_table(
            index="dataset", columns="method",
            values="compression_ratio", aggfunc="first",
        )
        print(pivot_cr.to_string(float_format="%8.2f"))
    except Exception as e:
        print(f"  (无法生成表格: {e})")

    print("\n" + "=" * 80)
    print("汇总 2：最大绝对误差 (max_abs_err)")
    print("=" * 80)
    try:
        pivot_err = df_ok.pivot_table(
            index="dataset", columns="method",
            values="max_abs_err", aggfunc="first",
        )
        print(pivot_err.to_string(float_format="%.6f"))
    except Exception as e:
        print(f"  (无法生成表格: {e})")

    print("\n" + "=" * 80)
    print("汇总 3：高能量区域 MSE (mse_high)")
    print("=" * 80)
    try:
        pivot_mse_h = df_ok.pivot_table(
            index="dataset", columns="method",
            values="mse_high", aggfunc="first",
        )
        print(pivot_mse_h.to_string(float_format="%.2e"))
    except Exception as e:
        print(f"  (无法生成表格: {e})")

    print("\n" + "=" * 80)
    print("汇总 4：低能量区域 MSE (mse_low)")
    print("=" * 80)
    try:
        pivot_mse_l = df_ok.pivot_table(
            index="dataset", columns="method",
            values="mse_low", aggfunc="first",
        )
        print(pivot_mse_l.to_string(float_format="%.2e"))
    except Exception as e:
        print(f"  (无法生成表格: {e})")

    print("\n" + "=" * 80)
    print("汇总 5：信噪比 SNR (dB)")
    print("=" * 80)
    try:
        pivot_snr = df_ok.pivot_table(
            index="dataset", columns="method",
            values="snr_db", aggfunc="first",
        )
        print(pivot_snr.to_string(float_format="%8.2f"))
    except Exception as e:
        print(f"  (无法生成表格: {e})")

    print("\n" + "=" * 80)
    print("汇总 6：编码时间 (秒)")
    print("=" * 80)
    try:
        pivot_enc = df_ok.pivot_table(
            index="dataset", columns="method",
            values="enc_time", aggfunc="first",
        )
        print(pivot_enc.to_string(float_format="%8.2f"))
    except Exception as e:
        print(f"  (无法生成表格: {e})")

    print("\n" + "=" * 80)
    print("汇总 7：解码时间 (秒)")
    print("=" * 80)
    try:
        pivot_dec = df_ok.pivot_table(
            index="dataset", columns="method",
            values="dec_time", aggfunc="first",
        )
        print(pivot_dec.to_string(float_format="%8.2f"))
    except Exception as e:
        print(f"  (无法生成表格: {e})")

    # 各方法跨数据集平均值
    print("\n" + "=" * 80)
    print("各方法跨数据集平均值")
    print("=" * 80)
    agg_cols = ["compression_ratio", "max_abs_err", "mse", "mse_high",
                "mse_low", "snr_db", "enc_time", "dec_time"]
    avail_cols = [c for c in agg_cols if c in df_ok.columns]
    if avail_cols:
        avg = df_ok.groupby("method")[avail_cols].mean()
        print(avg.to_string(float_format=lambda x: f"{x:.4e}"))

    print("\n完成。")


if __name__ == "__main__":
    main()
