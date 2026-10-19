"""
将 MiniSEED 文件转换为 HDF5 格式，用于基准测试。

选取 5 个具有不同信号特征的 Local Event 数据：
  1. event_0322_a: 704 ch, std=2593 (强事件)
  2. event_0322_b: 704 ch, std=1430 (中等事件)
  3. event_0620_a: 356 ch, std=1009 (中等)
  4. event_0620_b: 356 ch, std=141  (弱/噪声主导)
  5. event_0620_c: 286 ch, std=1127 (中等)

每个数据集提取前 6000 个时间样本，保持与 Alboran 数据集一致。
"""

import os
import numpy as np
import h5py
import obspy

# 源数据目录
MSEED_DIR = "/home/xdzhang/DAS_Compression/DASPack/daspack-main/Data_Read/data/Local_events"
OUTPUT_DIR = "/home/xdzhang/DAS_Compression/DASPack/improve/daspack-main/benchmark/data"

# 选取的 5 个数据集
SELECTED = [
    ("2022-03-22T17_41_40.MiDAS.mseed",        "event_0322_a"),
    ("2022-03-22T18_37_51.MiDAS.mseed",        "event_0322_b"),
    ("2022-06-20T01_05_08_ch0090_ch0445.MiDAS.mseed", "event_0620_a"),
    ("2022-06-20T01_39_43_ch0090_ch0445.MiDAS.mseed", "event_0620_b"),
    ("2022-06-20T03_03_09_ch0660_ch0945.MiDAS.mseed", "event_0620_c"),
]

N_SAMPLES = 6000  # 与 Alboran 数据集一致


def convert_mseed_to_h5(mseed_path, output_path, n_samples=N_SAMPLES):
    """将 MiniSEED 文件转换为 HDF5 格式。"""
    print(f"  读取: {os.path.basename(mseed_path)}")
    st = obspy.read(mseed_path)

    n_channels = len(st)
    sr = st[0].stats.sampling_rate
    npts = min(st[0].stats.npts, n_samples)

    # 提取数据为 2D 数组 (channels × time)
    data = np.zeros((n_channels, npts), dtype=np.float32)
    for i, tr in enumerate(st):
        data[i, :] = tr.data[:npts]

    # 转换为 float32 并标准化到合理的振幅范围
    # 原始数据是原始计数值，除以一个因子转换为类似 nanostrain 的尺度
    # 使用 std 归一化到 ~O(1) 量级，然后乘以 100 使范围类似 Alboran/Castor
    std = np.std(data)
    if std > 0:
        data = (data / std * 100.0).astype(np.float32)

    # 保存为 HDF5
    with h5py.File(output_path, "w") as f:
        dset = f.create_dataset("data", data=data)
        dset.attrs["dt_s"] = 1.0 / sr
        dset.attrs["begin_time"] = str(st[0].stats.starttime)
        dset.attrs["dx_m"] = 2.0  # 假设 2m 通道间距
        dset.attrs["n_channels"] = n_channels
        dset.attrs["n_samples"] = npts
        dset.attrs["sampling_rate"] = float(sr)
        dset.attrs["original_std"] = float(std)
        dset.attrs["source"] = os.path.basename(mseed_path)

    new_std = np.std(data)
    print(f"  输出: {n_channels} ch × {npts} samples, sr={sr:.0f} Hz")
    print(f"  原始 std={std:.2f}, 归一化后 std={new_std:.2f}")
    print(f"  数据范围: [{data.min():.4f}, {data.max():.4f}]")
    print(f"  保存到: {output_path}")

    return data


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for mseed_name, dataset_name in SELECTED:
        mseed_path = os.path.join(MSEED_DIR, mseed_name)
        output_path = os.path.join(OUTPUT_DIR, f"{dataset_name}.h5")

        if not os.path.exists(mseed_path):
            print(f"  文件不存在: {mseed_path}")
            continue

        print(f"\n--- {dataset_name} ---")
        convert_mseed_to_h5(mseed_path, output_path)

    # 验证所有 HDF5 文件
    print("\n" + "=" * 60)
    print("验证所有 HDF5 数据集:")
    print("=" * 60)

    h5_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".h5")])
    for fname in h5_files:
        path = os.path.join(OUTPUT_DIR, fname)
        with h5py.File(path, "r") as f:
            data = f["data"][:]
            print(f"  {fname}: {data.shape} {data.dtype}, "
                  f"std={np.std(data):.4f}, "
                  f"range=[{data.min():.2f}, {data.max():.2f}]")

    print(f"\n共 {len(h5_files)} 个数据集就绪。")
