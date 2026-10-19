"""
PhaseNet 相位拾取验证：对比原始 / DASPack / Adaptive2D 压缩后数据的相位拾取性能。

方法（与 STA/LTA 实验一致的自洽框架）：
  1. 在原始数据上运行 PhaseNet，得到参考 P/S 拾取（ground truth）。
  2. 对 DASPack 和 Adaptive2D 压缩-解压后的数据运行 PhaseNet。
  3. 将压缩后拾取与参考拾取按通道 + 时间容差匹配，计算
     Recall / Precision / F1 / 平均定时误差。

PhaseNet 通过 seisbench 加载预训练模型（ethz），在 GPU 上批量推理。
DAS 为单分量应变率数据，将同一通道复制为 3 分量输入。
每通道按自身标准差归一化（PhaseNet 训练约定）。

用法:
  /home/xdzhang/anaconda3/bin/python /home/xdzhang/DAS_Compression/GJI/code/phasenet_detection.py
"""

import os
import sys
import time
import numpy as np
import h5py
import pandas as pd
import torch
from scipy.signal import find_peaks

_HERE = os.path.dirname(os.path.abspath(__file__))
_DASPACK_SRC = os.path.join(_HERE, "daspack_src")
_DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "data"))
_RESULTS_DIR = os.path.normpath(os.path.join(_HERE, "..", "results"))

sys.path.insert(0, _DASPACK_SRC)
sys.path.insert(0, _HERE)

from sta_lta_detection import (  # noqa: E402
    daspack_compress_decompress,
    adaptive_compress_decompress,
)
import seisbench.models as sbm


# ======================================================================
# PhaseNet 推理与拾取
# ======================================================================

_PN_MODEL = None
_PN_SR = 100.0  # PhaseNet 训练采样率


def get_model():
    global _PN_MODEL
    if _PN_MODEL is None:
        m = sbm.PhaseNet.from_pretrained("ethz")
        m.eval()
        if torch.cuda.is_available():
            m = m.cuda()
        _PN_MODEL = m
    return _PN_MODEL


def run_phasenet(data, batch_size=256, p_threshold=0.3, s_threshold=0.3,
                 min_distance=50):
    """
    对二维 DAS 数据 (n_channels, n_samples) 运行 PhaseNet。

    Parameters
    ----------
    data : ndarray (n_channels, n_samples)
    batch_size : int
        GPU 批量大小。
    p_threshold, s_threshold : float
        P / S 拾取概率阈值。
    min_distance : int
        同一通道相邻拾取的最小样本间隔（去抖动）。

    Returns
    -------
    picks : list of dict
        每个 dict: {channel, sample, phase, prob}
    """
    model = get_model()
    n_ch, n_samp = data.shape
    device = next(model.parameters()).device

    # PhaseNet U-Net 对输入长度敏感：6000 样本可正常工作，而 12000
    # 会在 skip 连接处尺寸不匹配。因此沿时间轴切分为带重叠的窗口，
    # 拼接拾取结果并在重叠区去重（保留概率较高者）。
    win = 6000
    overlap = 500
    stride = win - overlap

    def _infer_block(sub):
        """sub: (B, T) -> pred: (B, 3, T)"""
        x = np.stack([sub, sub, sub], axis=1)  # (B, 3, T)
        std = np.std(x, axis=2, keepdims=True) + 1e-9
        x = x / std
        t = torch.from_numpy(x).to(device, non_blocking=True)
        with torch.no_grad():
            return model(t).cpu().numpy()  # (B, 3, T) -> [P, S, N]

    # 生成窗口起点（每个窗口长度均为 win，最后窗口对齐序列末尾）
    if n_samp <= win:
        win_starts = [0]
    else:
        win_starts = list(range(0, n_samp - win + 1, stride))
        if win_starts[-1] != n_samp - win:
            win_starts.append(n_samp - win)

    raw = []  # (channel, sample, phase, prob)
    for ws in win_starts:
        we = min(ws + win, n_samp)
        for start in range(0, n_ch, batch_size):
            end = min(start + batch_size, n_ch)
            sub = data[start:end, ws:we].astype(np.float32, copy=False)
            pred = _infer_block(sub)
            for i in range(end - start):
                ch = start + i
                p_prob = pred[i, 0]
                s_prob = pred[i, 1]
                pp, _ = find_peaks(p_prob, height=p_threshold,
                                   distance=min_distance)
                ps, _ = find_peaks(s_prob, height=s_threshold,
                                   distance=min_distance)
                for s in pp:
                    raw.append((ch, ws + int(s), "P", float(p_prob[s])))
                for s in ps:
                    raw.append((ch, ws + int(s), "S", float(s_prob[s])))

    # 重叠区去重：同通道同相位按时间排序，合并间隔 < min_distance 的拾取
    picks = []
    raw.sort(key=lambda r: (r[0], r[2], r[1]))
    last = {}  # (channel, phase) -> (sample, prob, index_in_picks)
    for ch, s, ph, pr in raw:
        key = (ch, ph)
        if key in last and s - last[key][0] < min_distance:
            if pr > last[key][1]:
                idx = last[key][2]
                picks[idx] = {"channel": ch, "sample": s, "phase": ph,
                              "prob": pr}
                last[key] = (s, pr, idx)
            continue
        picks.append({"channel": ch, "sample": s, "phase": ph, "prob": pr})
        last[key] = (s, pr, len(picks) - 1)
    return picks


# ======================================================================
# 拾取匹配
# ======================================================================

def match_picks(gt_picks, comp_picks, time_tol=30):
    """
    按通道和相位匹配压缩后拾取与参考拾取。

    匹配规则：同一通道、同一相位下，压缩拾取与最近的未匹配 GT 拾取
    的时间差 ≤ time_tol 则记为 TP。

    Returns
    -------
    tp, fp, fn, matched_pairs
    matched_pairs: list of (gt_sample, comp_sample, diff, phase)
    """
    def index(picks):
        d = {}
        for pk in picks:
            d.setdefault((pk["channel"], pk["phase"]), []).append(pk["sample"])
        for k in d:
            d[k].sort()
        return d

    gt_idx = index(gt_picks)
    comp_idx = index(comp_picks)

    tp = fp = fn = 0
    matched = []
    all_keys = set(gt_idx) | set(comp_idx)
    for key in all_keys:
        gt_times = gt_idx.get(key, [])
        comp_times = comp_idx.get(key, [])
        gt_used = [False] * len(gt_times)
        for ct in comp_times:
            best = -1
            best_d = time_tol + 1
            for j, gtt in enumerate(gt_times):
                if gt_used[j]:
                    continue
                d = abs(ct - gtt)
                if d <= time_tol and d < best_d:
                    best = j
                    best_d = d
            if best >= 0:
                gt_used[best] = True
                tp += 1
                matched.append((gt_times[best], ct, ct - gt_times[best], key[1]))
            else:
                fp += 1
        fn += sum(1 for u in gt_used if not u)
    return tp, fp, fn, matched


def metrics(tp, fp, fn, pairs):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    if pairs:
        diffs = np.abs([p[2] for p in pairs])
        mean_dt = float(np.mean(diffs))
        std_dt = float(np.std(diffs))
    else:
        mean_dt = std_dt = 0.0
    return prec, rec, f1, mean_dt, std_dt


# ======================================================================
# 数据加载
# ======================================================================

def load_dataset(name):
    fname = {
        "alboran": "7M_S_20231005-214004_RAW.h5",
        "castor": "ZI_C_20230609-102630_RAW.h5",
        "athens": "athens.h5",
    }[name]
    path = os.path.join(_DATA_DIR, fname)
    with h5py.File(path, "r") as f:
        data = np.array(f["data"][:]).astype(np.float32)
        sr = 1.0 / float(f["data"].attrs.get("dt_s", 0.01))
    return data, sr


def maybe_resample(data, sr, target=_PN_SR):
    """若采样率不是 100 Hz，按整数倍抽取到 100 Hz。"""
    if abs(sr - target) < 1e-3:
        return data
    factor = int(round(sr / target))
    if factor <= 0:
        return data
    return data[:, ::factor]


# ======================================================================
# 主流程
# ======================================================================

def evaluate_dataset(name, rel_tols, threads=4, batch_size=256,
                     time_tol=30):
    print(f"\n{'='*90}\n数据集: {name}\n{'='*90}")
    data, sr = load_dataset(name)
    data = maybe_resample(data, sr)
    n_ch, n_samp = data.shape
    data_std = float(np.std(data))
    print(f"  shape=({n_ch}, {n_samp})  sr={sr:.1f}Hz  std={data_std:.4f}  "
          f"duration={n_samp/_PN_SR:.1f}s")

    # ---- 原始数据参考拾取 ----
    print("  [PhaseNet] 原始数据参考拾取...", end=" ", flush=True)
    t0 = time.perf_counter()
    gt_picks = run_phasenet(data, batch_size=batch_size)
    print(f"{len(gt_picks)} picks ({time.perf_counter()-t0:.1f}s)")
    n_gt_p = sum(1 for p in gt_picks if p["phase"] == "P")
    n_gt_s = sum(1 for p in gt_picks if p["phase"] == "S")
    print(f"    P={n_gt_p}  S={n_gt_s}")

    rows = []
    for rel_tol in rel_tols:
        epsilon = rel_tol * data_std
        print(f"\n  rel_tol={rel_tol}  ε={epsilon:.4f}")

        # 原始数据行（GT 自比较）
        rows.append({
            "dataset": name, "rel_tol": rel_tol, "epsilon": epsilon,
            "method": "original",
            "n_picks": len(gt_picks), "n_p": n_gt_p, "n_s": n_gt_s,
            "tp": len(gt_picks), "fp": 0, "fn": 0,
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
            "mean_abs_dt": 0.0, "std_abs_dt": 0.0,
        })

        for method, fn in [("daspack", daspack_compress_decompress),
                           ("adaptive", adaptive_compress_decompress)]:
            print(f"    [{method}] 压缩-解压...", end=" ", flush=True)
            t0 = time.perf_counter()
            restored = fn(data, epsilon, threads)
            comp_t = time.perf_counter() - t0
            print(f"done ({comp_t:.1f}s)  拾取...", end=" ", flush=True)
            t0 = time.perf_counter()
            picks = run_phasenet(restored, batch_size=batch_size)
            print(f"{len(picks)} picks ({time.perf_counter()-t0:.1f}s)")
            tp, fp, fn_, pairs = match_picks(gt_picks, picks, time_tol)
            prec, rec, f1, mdt, sdt = metrics(tp, fp, fn_, pairs)
            print(f"      TP={tp} FP={fp} FN={fn_}  "
                  f"Recall={rec:.4f} F1={f1:.4f}  mean|Δt|={mdt:.2f}")
            rows.append({
                "dataset": name, "rel_tol": rel_tol, "epsilon": epsilon,
                "method": method,
                "n_picks": len(picks),
                "n_p": sum(1 for p in picks if p["phase"] == "P"),
                "n_s": sum(1 for p in picks if p["phase"] == "S"),
                "tp": tp, "fp": fp, "fn": fn_,
                "precision": prec, "recall": rec, "f1": f1,
                "mean_abs_dt": mdt, "std_abs_dt": sdt,
                "compress_time_s": comp_t,
            })
    return rows


def main():
    rel_tols = [0.001, 0.002, 0.005]
    # 命令行可指定数据集子集和输出文件名，例如:
    #   python phasenet_detection.py castor phasenet_castor.csv
    default_datasets = ["alboran", "athens"]
    valid = {"alboran", "castor", "athens"}
    cli = [a for a in sys.argv[1:] if not a.endswith(".csv")]
    datasets = [d for d in cli if d in valid] or default_datasets
    out_name = next((a for a in sys.argv[1:] if a.endswith(".csv")),
                    "phasenet_results.csv")
    threads = 4
    batch_size = 256
    time_tol = 30  # 0.3s at 100 Hz

    all_rows = []
    for name in datasets:
        all_rows.extend(evaluate_dataset(
            name, rel_tols, threads, batch_size, time_tol))

    os.makedirs(_RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(all_rows)
    out = os.path.join(_RESULTS_DIR, out_name)
    # 若使用自定义输出名且已有汇总 CSV，则合并去重后更新汇总文件
    df.to_csv(out, index=False)
    main_csv = os.path.join(_RESULTS_DIR, "phasenet_results.csv")
    if out_name != "phasenet_results.csv" and os.path.exists(main_csv):
        old = pd.read_csv(main_csv)
        merged = pd.concat([old, df], ignore_index=True)
        merged = merged.drop_duplicates(
            subset=["dataset", "rel_tol", "method"], keep="last")
        merged.to_csv(main_csv, index=False)
        print(f"已合并更新汇总文件 {main_csv}")
    print(f"\n结果已保存到 {out}")

    # 汇总
    print("\n" + "=" * 100)
    print("PhaseNet 相位拾取汇总 (Recall / F1 / mean|Δt|)")
    print("=" * 100)
    print(f"{'dataset':>10} {'rel_tol':>8} {'method':>10} "
          f"{'n_picks':>8} {'TP':>5} {'FP':>5} {'FN':>5} "
          f"{'Prec':>7} {'Rec':>7} {'F1':>7} {'|Δt|':>7}")
    print("-" * 100)
    for _, r in df.iterrows():
        if r["method"] == "original":
            continue
        print(f"{r['dataset']:>10} {r['rel_tol']:>8.4f} {r['method']:>10} "
              f"{int(r['n_picks']):>8d} {int(r['tp']):>5d} "
              f"{int(r['fp']):>5d} {int(r['fn']):>5d} "
              f"{r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f} "
              f"{r['mean_abs_dt']:>7.2f}")

    # 横向对比
    print("\n" + "=" * 100)
    print("横向对比 DASPack vs Adaptive2D")
    print("=" * 100)
    print(f"{'dataset':>10} {'rel_tol':>8} "
          f"{'DP Rec':>8} {'DP F1':>8} {'DP|Δt|':>8} "
          f"{'AD Rec':>8} {'AD F1':>8} {'AD|Δt|':>8} "
          f"{'ΔF1':>7}")
    print("-" * 100)
    for name in datasets:
        for rt in rel_tols:
            dp = df[(df.dataset == name) & (df.rel_tol == rt) &
                    (df.method == "daspack")]
            ad = df[(df.dataset == name) & (df.rel_tol == rt) &
                    (df.method == "adaptive")]
            if len(dp) and len(ad):
                dp, ad = dp.iloc[0], ad.iloc[0]
                print(f"{name:>10} {rt:>8.4f} "
                      f"{dp['recall']:>8.4f} {dp['f1']:>8.4f} "
                      f"{dp['mean_abs_dt']:>8.2f} "
                      f"{ad['recall']:>8.4f} {ad['f1']:>8.4f} "
                      f"{ad['mean_abs_dt']:>8.2f} "
                      f"{ad['f1']-dp['f1']:>+7.4f}")


if __name__ == "__main__":
    main()
