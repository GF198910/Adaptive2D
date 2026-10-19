import h5py, numpy as np, os

DATA_DIR = "/home/xdzhang/DAS_Compression/DASPack/improve/daspack-main/benchmark/data"

# 1. 9G_P: 取前 6000 列（与其他数据集一致）
src1 = "/home/xdzhang/DAS_Compression/DASPack/DASPack_2025_dataset/share_digital_csic/9G_P_20250606-115735_RAW.h5"
f1 = h5py.File(src1, 'r')
d1 = f1['data'][:, :6000].astype(np.float64)
f1.close()
print(f"9G_P: shape={d1.shape}, std={d1.std():.2f}")
# 保存
out1 = os.path.join(DATA_DIR, "9G_P.h5")
f1o = h5py.File(out1, 'w')
f1o.create_dataset('data', data=d1)
f1o.close()
print(f"Saved to {out1}")

# 2. Athens: 拼接前 5 个文件，取前 6000 列
athens_dir = "/home/xdzhang/DAS_Compression/DASPack/DASPack_2025_dataset/share_digital_csic/Athens_50Hz_20210923-070000"
files = sorted([f for f in os.listdir(athens_dir) if f.endswith('.h5')])
print(f"Athens files: {len(files)}")

# 读取第一个文件获取通道数
f2 = h5py.File(os.path.join(athens_dir, files[0]), 'r')
ds = f2['Acquisition']['Raw[0]']['RawData']
n_ch = ds.shape[0]
f2.close()

# 拼接前 5 个文件的时间维度
all_data = []
for fn in files[:5]:
    f = h5py.File(os.path.join(athens_dir, fn), 'r')
    d = f['Acquisition']['Raw[0]']['RawData'][:].astype(np.float64)
    all_data.append(d)
    f.close()

athens_data = np.concatenate(all_data, axis=1)  # 拼接时间维度
# 取前 6000 列
if athens_data.shape[1] > 6000:
    athens_data = athens_data[:, :6000]
print(f"Athens: shape={athens_data.shape}, std={athens_data.std():.2f}")

out2 = os.path.join(DATA_DIR, "athens.h5")
f2o = h5py.File(out2, 'w')
f2o.create_dataset('data', data=athens_data)
f2o.close()
print(f"Saved to {out2}")
