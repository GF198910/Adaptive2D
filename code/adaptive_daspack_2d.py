"""
2D Spatiotemporal Energy-Adaptive DASPack Compressor.

核心思想：
  用户指定最大误差容限 ε。
  - DASPack 原始方法：全局步长 = 2ε，最大误差 = ε
  - 自适应方法：
      高能量块：步长 = ε，  最大误差 = ε/2（更好保真度）
      低能量块：步长 = 2ε，最大误差 = ε （相同压缩效率）
      + 噪声阈值化：低能量块中 |x| < ε 的样本直接置零
        （置零误差 = |x| < ε，仍在容限内），大量零值显著提升熵编码效率
  两者都保证 max error ≤ ε，对比公平。

2D 时空分块的优势：
  地震事件在空间和时间上都是局部的。1D 分块（仅沿通道轴）会把
  整个时间段的噪声也归入高能量块。2D 分块能更精确地定位事件区域，
  减少使用细步长的数据量，从而在相同信号质量下获得更好的压缩率。

阈值模式：
  - "percentile"（默认）：按能量排序，top N% 的块为高能量。
    保证每个数据集都有高能量块，跨数据集一致性更好。
  - "factor"：块能量 > τ × 全局标准差 -> 高能量块。
    物理意义更明确，但可能对某些数据集不适用。
"""

import numpy as np
import pickle
import struct
from daspack import DASCoder, Quantizer


class AdaptiveDASPack2D:
    """
    2D 时空自适应量化 DASPack 压缩器。

    Parameters
    ----------
    tolerance : float
        最大绝对误差容限 ε。所有样本的重建误差 ≤ ε。
    block_channels : int
        每个块沿通道轴的大小。
    block_samples : int
        每个块沿时间轴的大小。
    threshold_mode : str, "percentile" 或 "factor"
        阈值确定方式。
    high_energy_fraction : float
        percentile 模式下，高能量块的比例（0-1），如 0.2 = top 20%。
    threshold_factor : float
        factor 模式下，块能量 > τ × 全局标准差 -> 高能量块。
    noise_thresholding : bool
        是否在低能量块中启用噪声阈值化（|x| < ε 的样本置零）。
    threads : int
        并行线程数。
    """

    def __init__(
        self,
        tolerance=0.1,
        block_channels=500,
        block_samples=500,
        threshold_mode="percentile",
        high_energy_fraction=0.2,
        threshold_factor=1.5,
        noise_thresholding=True,
        threads=4,
    ):
        self.name = "adaptive_daspack_2d"
        self.tolerance = tolerance
        self.block_channels = block_channels
        self.block_samples = block_samples
        self.threshold_mode = threshold_mode
        self.high_energy_fraction = high_energy_fraction
        self.threshold_factor = threshold_factor
        self.noise_thresholding = noise_thresholding
        self.threads = threads
        self.coder = DASCoder(threads=self.threads)

    def _compute_block_energies(self, data):
        """第一遍扫描：计算所有块的 RMS 能量。"""
        channels, samples = data.shape
        blocks = []

        for ch_start in range(0, channels, self.block_channels):
            for t_start in range(0, samples, self.block_samples):
                ch_end = min(ch_start + self.block_channels, channels)
                t_end = min(t_start + self.block_samples, samples)
                block = data[ch_start:ch_end, t_start:t_end]
                energy = float(np.sqrt(np.mean(block**2)))
                blocks.append({
                    "ch_range": (ch_start, ch_end),
                    "t_range": (t_start, t_end),
                    "energy": energy,
                })

        return blocks

    def _determine_threshold(self, blocks, data):
        """根据阈值模式确定高/低能量分界。"""
        energies = np.array([b["energy"] for b in blocks])
        global_std = float(np.std(data))

        if self.threshold_mode == "percentile":
            # top N% 的块为高能量
            n_high = max(1, int(len(blocks) * self.high_energy_fraction))
            sorted_energies = np.sort(energies)[::-1]  # 降序
            threshold = sorted_energies[n_high - 1] if n_high <= len(blocks) else sorted_energies[-1]
            is_high = energies >= threshold
        else:
            # factor 模式
            threshold = self.threshold_factor * global_std
            is_high = energies > threshold

        return is_high, global_std

    # ------------------------------------------------------------------
    # 编码
    # ------------------------------------------------------------------

    def _compress_impl(self, data, options=None):
        """压缩 2D DAS 数据。"""
        epsilon = self.tolerance
        if options is not None:
            if hasattr(options, "epsilon"):
                epsilon = options.epsilon
            elif hasattr(options, "quant_step"):
                epsilon = options.quant_step

        data = np.asarray(data, dtype=np.float64)
        channels, samples = data.shape

        # 第一遍：计算块能量
        blocks = self._compute_block_energies(data)

        # 确定阈值
        is_high, global_std = self._determine_threshold(blocks, data)

        # 第二遍：压缩每个块
        streams = []
        metadata = []
        n_high = 0
        n_low = 0

        for i, block_info in enumerate(blocks):
            ch_start, ch_end = block_info["ch_range"]
            t_start, t_end = block_info["t_range"]
            block = data[ch_start:ch_end, t_start:t_end].copy()

            if is_high[i]:
                step = epsilon
                block_type = "high"
                n_high += 1
            else:
                step = 2.0 * epsilon
                block_type = "low"
                n_low += 1
                # 噪声阈值化：低能量块中 |x| < ε 的样本置零
                # 置零误差 = |x| < ε ≤ ε，仍在容限内
                # 大量零值显著提升小波变换和熵编码的效率
                if self.noise_thresholding:
                    mask = np.abs(block) < epsilon
                    block[mask] = 0.0

            stream = self.coder.encode(block, Quantizer.Uniform(step=step))
            streams.append(stream)
            metadata.append({
                "step": step,
                "shape": (ch_end - ch_start, t_end - t_start),
                "len": len(stream),
                "ch_range": (ch_start, ch_end),
                "t_range": (t_start, t_end),
                "energy": block_info["energy"],
                "type": block_type,
                "thresholded": bool(self.noise_thresholding and block_type == "low"),
            })

        # 组装比特流
        global_meta = {
            "epsilon": epsilon,
            "shape": (channels, samples),
            "global_std": float(global_std),
            "global_mean": float(np.mean(data)),
            "block_channels": self.block_channels,
            "block_samples": self.block_samples,
            "threshold_mode": self.threshold_mode,
            "high_energy_fraction": self.high_energy_fraction,
            "threshold_factor": self.threshold_factor,
            "noise_thresholding": bool(self.noise_thresholding),
            "n_blocks": len(metadata),
            "n_high": n_high,
            "n_low": n_low,
        }

        meta_bytes = pickle.dumps({"global": global_meta, "blocks": metadata})

        result = bytearray()
        result.extend(struct.pack("<I", len(meta_bytes)))
        result.extend(meta_bytes)
        for stream in streams:
            result.extend(struct.pack("<I", len(stream)))
            result.extend(stream)

        return bytes(result)

    # ------------------------------------------------------------------
    # 解码
    # ------------------------------------------------------------------

    def _decompress_impl(self, compressed_stream):
        """解压比特流，返回 2D 数组。"""
        pos = 0
        meta_len = struct.unpack_from("<I", compressed_stream, pos)[0]
        pos += 4
        meta = pickle.loads(compressed_stream[pos:pos + meta_len])
        pos += meta_len

        global_meta = meta["global"]
        block_metas = meta["blocks"]

        channels, samples = global_meta["shape"]
        result = np.zeros((channels, samples), dtype=np.float64)

        for bm in block_metas:
            block_len = struct.unpack_from("<I", compressed_stream, pos)[0]
            pos += 4
            stream = compressed_stream[pos:pos + block_len]
            pos += block_len

            chunk = self.coder.decode(stream)

            ch_start, ch_end = bm["ch_range"]
            t_start, t_end = bm["t_range"]
            result[ch_start:ch_end, t_start:t_end] = chunk

        return result

    # ------------------------------------------------------------------
    # 基准测试框架接口
    # ------------------------------------------------------------------

    def write(self, arr, fn, options=None):
        compressed = self._compress_impl(arr, options)
        with open(fn, "wb") as f:
            f.write(compressed)

    def read(self, fn, options=None):
        with open(fn, "rb") as f:
            compressed = f.read()
        return self._decompress_impl(compressed)

    def compress(self, data, options=None):
        return self._compress_impl(data, options)

    def decompress(self, compressed_stream):
        return self._decompress_impl(compressed_stream)

    # ------------------------------------------------------------------
    # 分析辅助方法
    # ------------------------------------------------------------------

    def get_block_map(self, data, options=None):
        """返回每个块的分类信息，用于可视化和分析。"""
        data = np.asarray(data, dtype=np.float64)
        channels, samples = data.shape

        blocks = self._compute_block_energies(data)
        is_high, global_std = self._determine_threshold(blocks, data)

        block_map = np.zeros((channels, samples), dtype=np.int32)
        block_info = []

        epsilon = self.tolerance
        if options is not None:
            if hasattr(options, "epsilon"):
                epsilon = options.epsilon
            elif hasattr(options, "quant_step"):
                epsilon = options.quant_step

        for i, block_info_dict in enumerate(blocks):
            ch_start, ch_end = block_info_dict["ch_range"]
            t_start, t_end = block_info_dict["t_range"]
            high = bool(is_high[i])
            step = epsilon if high else 2.0 * epsilon

            block_map[ch_start:ch_end, t_start:t_end] = 1 if high else 0
            block_info.append({
                "ch_range": (ch_start, ch_end),
                "t_range": (t_start, t_end),
                "energy": block_info_dict["energy"],
                "is_high": high,
                "step": step,
            })

        return block_map, block_info
