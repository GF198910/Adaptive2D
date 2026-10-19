#!/usr/bin/env python3
"""
block_size_sweep.py

对 AdaptiveDASPack 进行 block_size 参数扫描，测试不同分块大小对压缩性能的影响。
使用方法：
    python block_size_sweep.py
结果将保存为 block_size_sweep_results.csv，并打印汇总表格。
"""

import time
import numpy as np
import h5py
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(__file__))
from my_daspack import AdaptiveDASPack

# ==================== 用户配置区域 ====================
DATA_FILES = {
    "alboran": "data/7M_S_20231005-214004_RAW.h5",
    "castor": "data/ZI_C_20230609-102630_RAW.h5",
}

# 要测试的 block_size 值列表（通道数）
BLOCK_SIZES = [50, 100, 150, 200, 250, 300, 350, 40 0, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]

# 固定参数
BASE_STEP = 0.1
THRESHOLD_FACTOR = 3.0
THREADS = 4

OUTPUT_CSV = "block_size_sweep_results.csv"
# =====================================================

def load_hdas_segment(h5_path):
    with h5py.File(h5_path, "r") as f:
        data_raw = np.array(f["data"][:]).astype(np.float32)
    return data_raw

def main():
    results = []

    for ds_name, file_path in DATA_FILES.items():
        print(f"\n=== 加载数据集 {ds_name} ===")
        data = load_hdas_segment(file_path)
        print(f"  形状: {data.shape}, dtype: {data.dtype}")

        data_f64 = data.astype(np.float64)

        for bs in BLOCK_SIZES:
            print(f"  测试 block_size = {bs} ...")

            coder = AdaptiveDASPack(
                base_step=BASE_STEP,
                block_size=bs,
                threshold_factor=THRESHOLD_FACTOR,
                threads=THREADS
            )

            t0 = time.perf_counter()
            compressed = coder.compress(data_f64)
            t_comp = time.perf_counter() - t0

            t0 = time.perf_counter()
            restored = coder.decompress(compressed)
            t_decomp = time.perf_counter() - t0

            original_bytes = data_f64.nbytes
            compressed_bytes = len(compressed)
            ratio = original_bytes / compressed_bytes

            max_err = np.max(np.abs(restored - data_f64))
            mse = np.mean((restored - data_f64)**2)

            tolerance = BASE_STEP
            verified = max_err <= tolerance + 1e-12

            print(f"     压缩比: {ratio:.3f}x, 最大误差: {max_err:.6f} (≤ {tolerance}), 验证: {verified}")
            print(f"     压缩时间: {t_comp:.3f}s, 解压时间: {t_decomp:.3f}s")

            results.append({
                'dataset': ds_name,
                'block_size': bs,
                'base_step': BASE_STEP,
                'threshold_factor': THRESHOLD_FACTOR,
                'original_bytes': original_bytes,
                'compressed_bytes': compressed_bytes,
                'compression_ratio': ratio,
                'max_abs_err': max_err,
                'mse': mse,
                'verified': verified,
                'compress_seconds': t_comp,
                'decompress_seconds': t_decomp
            })

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n结果已保存到 {OUTPUT_CSV}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()