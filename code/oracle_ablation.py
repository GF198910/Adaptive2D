"""
分类器消融（oracle 对比）：回应"percentile 不是事件检测器、只是固定比特重分配"的审稿风险。

逻辑：如果 percentile 能量分类能拿到 oracle（用原始数据上 STA/LTA 真实检出的
事件位置定义高能块）收益的绝大部分，那么"percentile 不是检测器"反而成为卖点——
零检测成本达到近 oracle 的精度分配效果。

对比配置（同一 ε、同一块网格 250x500、均保证 max|error| <= ε）：
  - percentile_p20    : 论文方法，top 20% 能量块为高能
  - oracle_free       : 含 STA/LTA 事件的块为高能（块数由事件决定）
  - percentile_matched: percentile，但高能块比例匹配 oracle 的比例
                        （等比特预算下检验"知道事件位置"是否仍更优）

指标：CR、整体 MSE、事件区 MSE（oracle 块内）、非事件区 MSE。

用法:
  /home/xdzhang/anaconda3/bin/python /home/xdzhang/DAS_Compression/GJI/code/oracle_ablation.py
"""

import os
import sys
import time
import numpy as np
import h5py
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "daspack_src"))
sys.path.insert(0, _HERE)

from adaptive_daspack_2d import AdaptiveDASPack2D  # noqa: E402
from sta_lta_detection import detect_events        # noqa: E402

_DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "data"))
_RESULTS_DIR = os.path.normpath(os.path.join(_HERE, "..", "results"))

DATASETS = [
    "7M_S_20231005-214004_RAW", "ZI_C_20230609-102630_RAW", "9G_P", "athens",
    "event_0322_a", "event_0322_b", "event_0620_a", "event_0620_b", "event_0620_c",
]

REL_TOLS = [0.001, 0.005]
BLOCK_CH, BLOCK_T = 250, 500
FRAC = 0.2
THREADS = 4
# STA/LTA 参数与论文一致
STA, LTA, THR, DEB = 50, 500, 3.0, 500


class MaskAdaptive(AdaptiveDASPack2D):
    """外部提供高能块掩码的 Adaptive2D（跳过内部能量分类）。"""

    def __init__(self, *args, mask=None, **kw):
        super().__init__(*args, **kw)
        self.mask = np.asarray(mask, dtype=bool)

    def _determine_threshold(self, blocks, data):
        assert len(self.mask) == len(blocks)
        return self.mask.copy(), float(np.std(data))


def load(name):
    with h5py.File(os.path.join(_DATA_DIR, f"{name}.h5"), "r") as f:
        return np.array(f["data"][:]).astype(np.float32)


def oracle_mask(arr, triggers, bc=BLOCK_CH, bt=BLOCK_T):
    """按 _compute_block_energies 的遍历顺序生成布尔掩码：块内含任一触发即为高能。"""
    C, T = arr.shape
    nbc = int(np.ceil(C / bc))
    mask = np.zeros(nbc * int(np.ceil(T / bt)), dtype=bool)
    for ch, t in triggers:
        mask[(ch // bc) * int(np.ceil(T / bt)) + (t // bt)] = True
    return mask


def region_metrics(arr, rec, oracle_blocks):
    d = rec.astype(np.float64) - arr.astype(np.float64)
    ev = oracle_blocks
    return {
        "max_abs_err": float(np.max(np.abs(d))),
        "mse": float(np.mean(d ** 2)),
        "mse_event": float(np.mean(d[ev] ** 2)) if ev.any() else 0.0,
        "mse_nonevent": float(np.mean(d[~ev] ** 2)) if (~ev).any() else 0.0,
    }


def main():
    rows = []
    for name in DATASETS:
        arr = load(name)
        C, T = arr.shape
        std = float(np.std(arr))
        orig_bytes = arr.nbytes
        print(f"\n>>> {name} shape=({C},{T}) std={std:.3f}")

        # 1) 原始数据 GT 事件检测
        t0 = time.perf_counter()
        triggers, _ = detect_events(arr, STA, LTA, THR, DEB)
        print(f"  GT events: {len(triggers)} ({time.perf_counter()-t0:.1f}s)")

        # 2) oracle 掩码（块级）及其像素级版本
        omask = oracle_mask(arr, triggers)
        oracle_frac = float(omask.mean())
        omap = np.zeros((C, T), dtype=bool)
        idx = 0
        for cs in range(0, C, BLOCK_CH):
            for ts in range(0, T, BLOCK_T):
                if omask[idx]:
                    omap[cs:cs + BLOCK_CH, ts:ts + BLOCK_T] = True
                idx += 1
        print(f"  oracle high-energy blocks: {omask.sum()}/{len(omask)} "
              f"(frac={oracle_frac:.3f})")

        configs = {
            "percentile_p20": dict(kind="pct", frac=FRAC),
            "oracle_free": dict(kind="mask", mask=omask),
            "percentile_matched": dict(kind="pct", frac=max(oracle_frac, 1.0 / len(omask))),
        }

        for rel in REL_TOLS:
            eps = rel * std
            for cname, cfg in configs.items():
                t0 = time.perf_counter()
                if cfg["kind"] == "pct":
                    codec = AdaptiveDASPack2D(
                        tolerance=eps, block_channels=BLOCK_CH,
                        block_samples=BLOCK_T, threshold_mode="percentile",
                        high_energy_fraction=cfg["frac"],
                        noise_thresholding=True, threads=THREADS)
                else:
                    codec = MaskAdaptive(
                        tolerance=eps, block_channels=BLOCK_CH,
                        block_samples=BLOCK_T, noise_thresholding=True,
                        threads=THREADS, mask=cfg["mask"])
                buf = codec.compress(arr)
                rec = codec.decompress(buf)
                dt = time.perf_counter() - t0
                m = region_metrics(arr, rec, omap)
                cr = orig_bytes / len(buf)
                rows.append({"dataset": name, "config": cname, "rel_tol": rel,
                             "eps": eps, "cr": cr, "oracle_frac": oracle_frac,
                             "gt_events": len(triggers), "enc_time_s": dt, **m})
                print(f"  [{cname:18s}] rel={rel:.4f} CR={cr:7.2f} "
                      f"mse={m['mse']:.3e} mse_event={m['mse_event']:.3e} ({dt:.1f}s)")

    df = pd.DataFrame(rows)
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    out = os.path.join(_RESULTS_DIR, "oracle_ablation_results.csv")
    df.to_csv(out, index=False)
    print(f"\n已保存 {out}")

    # 汇总：同 rel_tol 下，percentile_matched vs oracle_free 的事件区 MSE 比
    print("\n" + "=" * 100)
    print("消融汇总：mse_event 与 CR（rel_tol=0.005 为主）")
    print("=" * 100)
    piv = df[df.rel_tol == REL_TOLS[-1]].pivot_table(
        index="dataset", columns="config",
        values=["cr", "mse", "mse_event"])
    print(piv.to_string(float_format="%.3e"))


if __name__ == "__main__":
    main()
