"""
最小可行实验：DCT 分块变换编码（有界误差的变换域量化）用于 DAS 数据压缩。

目的
----
验证"从图像角度设计 DAS 专用压缩"的路线是否可行：在相同的最大误差上界
max|error| <= eps 下，DCT 分块变换编码能否达到与 DASPack（小波 + 均匀量化）
相当的压缩率。

方法（有界误差的变换域量化，MVP 版）
------------------------------------
  1. 将 (通道 x 时间) 数据划分为 N x N 的二维块（默认 16x16），边缘 replicate 填充。
  2. 对每个块做正交 2D DCT-II（scipy, norm='ortho'）。
  3. 阈值编码（Gonzalez 8.9）：把空间贡献小的系数置零，截断误差预算 B_tr = eps/2。
  4. 保留系数量化：步长 Delta_uv = q * A_uv，量化误差预算 B_q = eps/2。
  5. 有界保证（l-inf 加权和界）：
        max|error| <= sum_{零} |C_uv| * W_uv          (<= eps/2)
                    + sum_{保留} (Delta_uv/2) * W_uv   (<= eps/2)
     其中 W_uv = ||phi_u||_inf * ||phi_v||_inf 为正交 DCT 基的 l-inf 范数乘积。
     取 q = 2*B_q / sum_{保留} A_uv * W_uv，使第二项恰为 eps/2，故总误差 <= eps。
  6. 熵编码：zigzag 扫描 + zlib（MVP 用简单后端，便于隔离"变换本身"的贡献）。

A_uv 分配矩阵：
  - 'l1inf'  : A = W（由变换几何决定的中性分配）
  - 'seismic': A = W * M，M 为地震先验（时间频率越高，量化步长越大）

对比基线（相同 eps）
--------------------
  - DASPack : 样本域均匀量化 step=2eps（max error=eps）+ 小波/LPC/熵编码
  - Gzip    : 样本域均匀量化 step=2eps + zlib（与本文同一 zlib 后端，隔离变换贡献）

用法:
  /home/xdzhang/anaconda3/bin/python /home/xdzhang/DAS_Compression/GJI/code/dct_mvp.py
"""

import os
import sys
import time
import zlib
import struct
import pickle
import numpy as np
import h5py
import pandas as pd
from scipy.fft import dctn, idctn

# ---------------------------------------------------------------------
# 路径设置：本地 daspack 包
# ---------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_DASPACK_SRC = os.path.join(_HERE, "daspack_src")
_DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "data"))
_RESULTS_DIR = os.path.normpath(os.path.join(_HERE, "..", "results"))

sys.path.insert(0, _DASPACK_SRC)
sys.path.insert(0, _HERE)

from daspack import DASCoder, Quantizer  # noqa: E402


# =====================================================================
# 数据集
# =====================================================================
DATASETS = [
    "7M_S_20231005-214004_RAW",   # alboran
    "ZI_C_20230609-102630_RAW",   # castor
    "9G_P",
    "athens",
    "event_0322_a",
    "event_0322_b",
    "event_0620_a",
    "event_0620_b",
    "event_0620_c",
]


def load_dataset(name):
    """加载数据集为 float32 二维数组（与既有脚本一致）。"""
    path = os.path.join(_DATA_DIR, f"{name}.h5")
    with h5py.File(path, "r") as f:
        return np.array(f["data"][:]).astype(np.float32)


# =====================================================================
# 正交 DCT 基的 l-inf 范数与 zigzag 顺序
# =====================================================================
def dct_basis_linf_1d(N):
    """1D 正交 DCT-II 每个基向量的 l-inf 范数（max_n |phi_k(n)|）。"""
    E = np.eye(N)
    B = dctn(E, type=2, norm="ortho", axes=(0,))   # B[k, n] = phi_k(n)
    return np.max(np.abs(B), axis=1)               # (N,)


def zigzag_indices(N):
    """返回 zigzag 扫描顺序的 (ii, jj) 索引数组。"""
    ii, jj = [], []
    for s in range(2 * N - 1):
        rng = range(max(0, s - N + 1), min(s, N - 1) + 1)
        if s % 2 == 0:
            for i in rng:
                ii.append(i); jj.append(s - i)
        else:
            for i in reversed(list(rng)):
                ii.append(i); jj.append(s - i)
    return np.array(ii), np.array(jj)


def make_alloc_matrix(N, mode="l1inf", alpha=1.0, gamma=1.0):
    """
    分配矩阵 A（决定每个系数的量化步长 Delta = q*A）。
      - W = W_uv = ||phi_u||_inf * ||phi_v||_inf（几何权重）
      - mode='l1inf'  : A = W
      - mode='seismic': A = W * M, M = 1 + alpha*(v/(N-1))^gamma（v = 时间频率索引）
    """
    w1 = dct_basis_linf_1d(N)
    W = np.outer(w1, w1)                     # (N, N): axis0=通道频率 u, axis1=时间频率 v
    if mode == "seismic":
        v = np.arange(N) / (N - 1)
        M = 1.0 + alpha * (v ** gamma)       # 时间频率越高 -> 越大
        A = W * M[None, :]
    else:
        A = W.copy()
    return W, A


# =====================================================================
# DCT 分块编解码器
# =====================================================================
class DCTBlockCodec:
    """
    DCT 分块变换编码，保证 max|error| <= eps。

    Parameters
    ----------
    N : int
        DCT 块尺寸（N x N）。
    alloc_mode : str
        'l1inf' 或 'seismic'。
    alpha, gamma : float
        seismic 模式的先验参数。
    chunk_blocks : int
        分块处理的最大块数（控制内存）。
    """

    def __init__(self, N=16, alloc_mode="l1inf", alpha=1.0, gamma=1.0,
                 chunk_blocks=20000):
        self.N = N
        self.alloc_mode = alloc_mode
        self.alpha = alpha
        self.gamma = gamma
        self.chunk_blocks = chunk_blocks
        self.W, self.A = make_alloc_matrix(N, alloc_mode, alpha, gamma)
        self._ii, self._jj = zigzag_indices(N)
        self._zz_idx = self._ii * N + self._jj          # 展平索引（行优先）

    # ---- 内核：对一个块批次编码 ----
    def _encode_chunk(self, blocks, eps):
        """
        blocks: (nb, N, N) float64
        返回 (cq_int32 沿 zigzag 展平的 (nb, N*N), q (nb,))
        """
        nb = blocks.shape[0]
        N = self.N
        C = dctn(blocks, type=2, norm="ortho", axes=(-2, -1))
        contrib = np.abs(C) * self.W                      # (nb,N,N) 丢该系数的最大空间贡献

        cf = contrib.reshape(nb, -1)
        order = np.argsort(cf, axis=1)                    # 升序
        cf_sorted = np.take_along_axis(cf, order, axis=1)
        cum = np.cumsum(cf_sorted, axis=1)
        B_tr = eps / 2.0
        keep_sorted = cum > B_tr                          # 累计贡献 <= Btr 的被置零
        keep_flat = np.zeros_like(keep_sorted)
        np.put_along_axis(keep_flat, order, keep_sorted, axis=1)
        keep = keep_flat.reshape(nb, N, N)

        # 量化预算：sum_keep (Delta/2)*W = (q/2)*sum_keep A*W <= B_q
        denom = (keep * (self.A * self.W)).sum(axis=(1, 2))
        q = np.where(denom > 0, (2.0 * (eps / 2.0)) / np.maximum(denom, 1e-300), 1.0)

        Delta = q[:, None, None] * self.A[None]
        Cq = np.round(C / Delta)
        Cq = np.where(keep, Cq, 0.0)

        # 理论界自检：截断预算 + 量化预算 <= eps
        trunc_err = (contrib * (~keep)).sum(axis=(1, 2))
        quant_bound = ((Delta * 0.5) * self.W * keep).sum(axis=(1, 2))
        total_bound = trunc_err + quant_bound
        assert np.all(total_bound <= eps + 1e-6), \
            f"bound violated: max={total_bound.max()} eps={eps}"

        cq_zig = Cq.reshape(nb, -1)[:, self._zz_idx].astype(np.int32)
        return cq_zig, q.astype(np.float64)

    def compress(self, data, eps):
        data = np.asarray(data, dtype=np.float64)
        C, T = data.shape
        N = self.N
        Cpad = int(np.ceil(C / N)) * N
        Tpad = int(np.ceil(T / N)) * N
        padded = np.pad(data, ((0, Cpad - C), (0, Tpad - T)), mode="edge")

        # 重排为 (nbc, nbt, N, N) -> 展平 (nb, N, N)
        blk = padded.reshape(Cpad // N, N, Tpad // N, N).transpose(0, 2, 1, 3)
        blk = np.ascontiguousarray(blk.reshape(-1, N, N))

        nb = blk.shape[0]
        cq_parts, q_parts = [], []
        for s in range(0, nb, self.chunk_blocks):
            cqc, qc = self._encode_chunk(blk[s:s + self.chunk_blocks], eps)
            cq_parts.append(cqc)
            q_parts.append(qc)
        cq = np.concatenate(cq_parts, axis=0)
        q = np.concatenate(q_parts, axis=0)

        # 熵编码：zigzag 后的系数流（int32）+ zlib；元数据记录每块的 q
        coef_bytes = zlib.compress(cq.tobytes(), level=6)
        meta = {
            "shape": (C, T),
            "Cpad": Cpad, "Tpad": Tpad,
            "N": N, "eps": eps,
            "alloc_mode": self.alloc_mode,
            "alpha": self.alpha, "gamma": self.gamma,
            "nb": nb,
            "q": q,
        }
        meta_bytes = pickle.dumps(meta)
        out = bytearray()
        out.extend(struct.pack("<I", len(meta_bytes)))
        out.extend(meta_bytes)
        out.extend(struct.pack("<I", len(coef_bytes)))
        out.extend(coef_bytes)
        return bytes(out)

    def _decode_chunk(self, cq_zig, q):
        nb = cq_zig.shape[0]
        N = self.N
        Cq = np.zeros((nb, N * N), dtype=np.float64)
        Cq[:, self._zz_idx] = cq_zig.astype(np.float64)
        Cq = Cq.reshape(nb, N, N)
        Delta = q[:, None, None] * self.A[None]
        C_hat = Cq * Delta
        return idctn(C_hat, type=2, norm="ortho", axes=(-2, -1))

    def decompress(self, blob):
        pos = 0
        meta_len = struct.unpack_from("<I", blob, pos)[0]; pos += 4
        meta = pickle.loads(blob[pos:pos + meta_len]); pos += meta_len
        coef_len = struct.unpack_from("<I", blob, pos)[0]; pos += 4
        coef_bytes = blob[pos:pos + coef_len]; pos += coef_len

        N = meta["N"]
        nb = meta["nb"]
        q = np.asarray(meta["q"], dtype=np.float64)
        cq = np.frombuffer(zlib.decompress(coef_bytes), dtype=np.int32).reshape(nb, N * N)

        Cpad, Tpad = meta["Cpad"], meta["Tpad"]
        out = np.zeros((nb, N, N), dtype=np.float64)
        for s in range(0, nb, self.chunk_blocks):
            out[s:s + self.chunk_blocks] = self._decode_chunk(
                cq[s:s + self.chunk_blocks], q[s:s + self.chunk_blocks])
        # 还原 (nbc, nbt, N, N) -> (Cpad, Tpad)
        grid = out.reshape(Cpad // N, Tpad // N, N, N).transpose(0, 2, 1, 3)
        full = grid.reshape(Cpad, Tpad)
        C, T = meta["shape"]
        return full[:C, :T]
        # 注：Cpb 未使用，保留以备扩展


# =====================================================================
# 基线
# =====================================================================
def gzip_sample_quant_decompress(arr, eps):
    """样本域均匀量化 step=2eps + zlib（与 DCT 共用 zlib 后端）。"""
    step = 2.0 * eps
    qi = np.round(arr.astype(np.float64) / step).astype(np.int32)
    buf = zlib.compress(qi.tobytes(), level=6)
    restored = np.frombuffer(zlib.decompress(buf), dtype=np.int32).reshape(arr.shape) * step
    return len(buf), restored


def daspack_encode_decompress(arr, eps, threads=4):
    coder = DASCoder(threads=threads)
    buf = coder.encode(arr.astype(np.float64), Quantizer.Uniform(step=2.0 * eps))
    restored = coder.decode(buf)
    return len(buf), restored


# =====================================================================
# 指标
# =====================================================================
def metrics(original, restored):
    d = restored.astype(np.float64) - original.astype(np.float64)
    mse = float(np.mean(d ** 2))
    sp = float(np.mean(original.astype(np.float64) ** 2))
    snr = 10 * np.log10(sp / mse) if mse > 0 else float("inf")
    return {"max_abs_err": float(np.max(np.abs(d))), "mse": mse,
            "rmse": float(np.sqrt(mse)), "snr_db": snr}


# =====================================================================
# 主流程
# =====================================================================
def main():
    rel_tols = [0.001, 0.005]
    N = 16
    threads = 4

    codecs = {
        "dct_l1inf": DCTBlockCodec(N=N, alloc_mode="l1inf"),
        "dct_seismic": DCTBlockCodec(N=N, alloc_mode="seismic", alpha=1.0, gamma=1.0),
    }

    rows = []
    print("=" * 118)
    print("DCT 分块变换编码 最小可行实验（相对容限 rel_tol = eps / std）")
    print(f"  DCT 块 {N}x{N} | 熵编码 zlib | 对比基线 Gzip(样本域+zlib) 与 DASPack")
    print("=" * 118)

    for name in DATASETS:
        arr = load_dataset(name)
        C, T = arr.shape
        std = float(np.std(arr))
        orig_bytes = arr.nbytes   # float32
        print(f"\n>>> {name}  shape=({C},{T})  std={std:.4f}  orig={orig_bytes/1e6:.1f}MB")

        for rel_tol in rel_tols:
            eps = rel_tol * std
            base = {"dataset": name, "shape": f"{C}x{T}", "std": std,
                    "rel_tol": rel_tol, "eps": eps}

            # --- DCT (两种分配矩阵) ---
            for cname, codec in codecs.items():
                t0 = time.perf_counter()
                blob = codec.compress(arr, eps)
                enc_t = time.perf_counter() - t0
                rec = codec.decompress(blob)
                cr = orig_bytes / len(blob)
                m = metrics(arr, rec)
                rows.append({**base, "method": cname, "cr": cr,
                             "compressed_MB": len(blob) / 1e6,
                             "enc_time_s": enc_t, **m})
                print(f"    [{cname:12s}] CR={cr:6.2f}  MaxAE={m['max_abs_err']:.4f} "
                      f"(eps={eps:.4f})  MSE={m['mse']:.3e}  SNR={m['snr_db']:5.1f}dB  "
                      f"enc={enc_t:.1f}s")

            # --- Gzip 基线 ---
            nb, rec = gzip_sample_quant_decompress(arr, eps)
            cr = orig_bytes / nb
            m = metrics(arr, rec)
            rows.append({**base, "method": "gzip_zlib", "cr": cr,
                         "compressed_MB": nb / 1e6, "enc_time_s": float("nan"), **m})
            print(f"    [{'gzip_zlib':12s}] CR={cr:6.2f}  MaxAE={m['max_abs_err']:.4f}  "
                  f"MSE={m['mse']:.3e}  SNR={m['snr_db']:5.1f}dB")

            # --- DASPack 基线 ---
            t0 = time.perf_counter()
            nb, rec = daspack_encode_decompress(arr, eps, threads)
            enc_t = time.perf_counter() - t0
            cr = orig_bytes / nb
            m = metrics(arr, rec)
            rows.append({**base, "method": "daspack", "cr": cr,
                         "compressed_MB": nb / 1e6, "enc_time_s": enc_t, **m})
            print(f"    [{'daspack':12s}] CR={cr:6.2f}  MaxAE={m['max_abs_err']:.4f}  "
                  f"MSE={m['mse']:.3e}  SNR={m['snr_db']:5.1f}dB  enc={enc_t:.1f}s")

    df = pd.DataFrame(rows)
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    out_csv = os.path.join(_RESULTS_DIR, "dct_mvp_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n结果已保存到 {out_csv}")

    # 汇总：各方法 CR（按数据集 x rel_tol）
    print("\n" + "=" * 118)
    print("CR 汇总（行=数据集, 列=方法, 按 rel_tol 分组）")
    print("=" * 118)
    for rel_tol in rel_tols:
        sub = df[df["rel_tol"] == rel_tol]
        print(f"\n--- rel_tol = {rel_tol} ---")
        piv = sub.pivot_table(index="dataset", columns="method", values="cr")
        cols = [c for c in ["dct_l1inf", "dct_seismic", "gzip_zlib", "daspack"] if c in piv.columns]
        print(piv[cols].to_string(float_format="%.2f"))

    # 相对 DASPack 的 CR 比值
    print("\n" + "=" * 118)
    print("相对 DASPack 的 CR 比值（DCT / DASPack，越接近 1 越好）")
    print("=" * 118)
    for rel_tol in rel_tols:
        sub = df[df["rel_tol"] == rel_tol]
        piv = sub.pivot_table(index="dataset", columns="method", values="cr")
        if "daspack" in piv.columns:
            for m in ["dct_l1inf", "dct_seismic", "gzip_zlib"]:
                if m in piv.columns:
                    ratio = (piv[m] / piv["daspack"]).mean()
                    print(f"  rel_tol={rel_tol}  {m:12s}: mean CR ratio vs DASPack = {ratio:.3f}")


if __name__ == "__main__":
    main()
