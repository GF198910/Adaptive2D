# Adaptive2D: Spatiotemporal Energy-Adaptive Quantization for Bounded-Error DAS Compression

Replication package for the manuscript

> Xinda Zhang and Hang Gao, "Spatiotemporal Energy-Adaptive Quantization for Bounded-Error
> Compression of Distributed Acoustic Sensing Data", *Geophysical Journal International*
> (submitted, 2026).

This archive contains the Adaptive2D codec wrapper, the baseline codecs, and every
script needed to reproduce all tables and figures of the paper, together with the
precomputed result files (CSV) that the published tables were built from.

Release: https://github.com/GF198910/Adaptive2D/releases/tag/v1.2.0
Source repository: https://github.com/GF198910/Adaptive2D

---

## 1. Repository layout

```
.
├── README.md                  ← this file
├── requirements.txt           ← Python dependencies
├── code/                      ← all source code (run scripts from here)
│   ├── adaptive_daspack_2d.py       # Adaptive2D codec (the proposed method)
│   ├── compression_methods.py       # codec abstractions + ZFP / JPEG2000 baselines
│   ├── signal_evaluation.py         # metrics (MSE / SNR / MaxAE, region masks)
│   ├── daspack_src/                 # DASPack engine snapshot (see §4)
│   │
│   ├── fair_benchmark.py            # → Table 2 (DASPack rows), Table 3
│   ├── extended_benchmark.py        # → Table 2 (extended datasets)
│   ├── benchmark_new_datasets.py    # → Table 2 (MiDAS datasets)
│   ├── comprehensive_benchmark.py   # consolidated benchmark → Table 2, Table 5 inputs
│   ├── rate_distortion.py           # → Figure 1 (Alboran & Castor RD curves)
│   ├── rd_all.py                    # → Table 4 (rate-matched analysis, 9 datasets)
│   ├── oracle_ablation.py           # → Table 10 (oracle classifier ablation)
│   ├── param_sensitivity.py         # → Figure 4 (block size & fraction sweeps)
│   ├── block_size_sweep.py          # → Figure 4 inputs
│   ├── sta_lta_detection.py         # → Table 6 (STA/LTA on Alboran)
│   ├── sta_lta_midas.py             # → Table 7 (STA/LTA on five MiDAS datasets)
│   ├── phasenet_detection.py        # → Table 8 (PhaseNet phase picking; needs seisbench)
│   ├── step_ratio_sweep.py          # → Table 9 (step ratio ρ sweep)
│   ├── dct_mvp.py                   # → §5.6 bounded-error block-DCT baseline
│   │
│   ├── gen_fig2_waveform.py          # → Figure 2
│   ├── gen_fig3_v2.py               # → Figure 3 (error maps)
│   ├── gen_pub_figures.py           # → Figure 5 (summary)
│   ├── combine_figures.py           # figure post-processing helper
│   ├── gen_fig6_sta_lta.py          # → Figure 6
│   ├── waveform_visualization.py    # waveform plotting helpers
│   ├── prepare_datasets.py          # data conversion helpers (optional; needs obspy)
│   └── prepare_new_datasets.py      # data conversion helpers
│
├── data/                      ← datasets are NOT included; see §3
│   └── README.md              ← download links and expected file names
├── results/                   ← precomputed CSV outputs used in the paper
└── figures/                   ← the published figures (reference outputs)
```

All analysis scripts resolve paths relative to `code/`, expecting `data/` and
`results/` as sibling directories. Run every script from the `code/` directory.

## 2. Installation

Tested with Python 3.12 (Anaconda) on Linux x86-64.

```bash
pip install -r requirements.txt

# Build and install the bundled DASPack engine (Rust / PyO3, needs a Rust toolchain:
# https://rustup.rs). This compiles the `daspack` Python extension in place:
pip install ./code/daspack_src
```

`obspy` is only needed by `prepare_datasets.py` (data conversion from raw formats);
it is not required to reproduce any table or figure from the released HDF5 files.

## 3. Data

The 9 HDF5 datasets (≈1 GB total) are **not** redistributed with this archive.

- **Alboran, Castor, Rhonegletscher (9G_P), Athens** — distributed with the DASPack
  software release: https://github.com/asleix/daspack (see `data/README.md` for the
  exact file names this package expects).
- **MiDAS local-earthquake recordings** (event_0322_a/b, event_0620_a/b/c) —
  available from the Taiwan Academia Sinica MiDAS project upon reasonable request;
  we do not have redistribution rights.

Place the nine `.h5` files in `data/`. Each file must contain a dataset named
`data` of shape `(n_channels, n_samples)`, float32.

## 4. DASPack engine

`code/daspack_src/` is an unmodified main-branch snapshot of DASPack
(https://github.com/asleix/daspack, v0.0.1a0, downloaded May 2026),
included so that the archive is self-contained and bit-for-bit reproducible.
DASPack is © its authors and distributed under its original licence
(see `code/daspack_src/LICENSE`). **No modifications were made to the DASPack core.**
If you prefer the upstream version, `pip install daspack` also works, but minor
numerical differences relative to the released CSVs may occur across versions.

## 5. Reproducing the paper

Run from `code/`. Every script writes CSV(s) into `results/` (existing CSVs are
the released reference outputs; re-running overwrites them). Typical wall times on
a desktop CPU: minutes per script, except the full benchmarks (~1 h).

| Paper artifact | Command | Output CSV(s) |
|---|---|---|
| Table 2 (main results, 9 datasets) | `python comprehensive_benchmark.py` | `comprehensive_benchmark_results.csv` (+ `fair_benchmark_results.csv`, `extended_benchmark_results.csv`, `new_datasets_results.csv`) |
| Table 3 (ZFP / JPEG2000 comparison) | `python fair_benchmark.py` | `fair_benchmark_results.csv` |
| Table 4 (rate-matched analysis) | `python rd_all.py` | `rd_all_results.csv`, `rd_all_rate_matched.csv` |
| Table 5 (statistics) | derived from the Table 2 CSVs (Wilcoxon signed-rank, n = 9) | `statistical_analysis.csv` |
| Table 6 (STA/LTA, Alboran) | `python sta_lta_detection.py` | `sta_lta_results.csv`, `sta_lta_multi_tolerance.csv` |
| Table 7 (STA/LTA, MiDAS) | `python sta_lta_midas.py` | `sta_lta_midas_results.csv` |
| Table 8 (PhaseNet picking) | `python phasenet_detection.py` | `phasenet_results.csv` |
| Table 9 (step ratio sweep) | `python step_ratio_sweep.py` | `step_ratio_sweep.csv` |
| Table 10 (oracle ablation) | `python oracle_ablation.py` | `oracle_ablation_results.csv` |
| Figure 1 (RD curves) | `python rate_distortion.py` | `rate_distortion_results.csv` → `fig1_rate_distortion.png` |
| Figure 2 (waveforms) | `python gen_fig2_waveform.py` | `fig2_waveform.png` |
| Figure 3 (error maps) | `python gen_fig3_v2.py` | `fig3_error_map.png` |
| Figure 4 (sensitivity) | `python param_sensitivity.py`, `python block_size_sweep.py` | `sweep_block_size.csv`, `sweep_fraction.csv`, `threshold_sweep_results.csv`, `block_size_sweep_results.csv` |
| Figure 5 (summary) | `python gen_pub_figures.py` | `fig5_summary.png` |
| Figure 6 (STA/LTA illustration) | `python gen_fig6_sta_lta.py` | `fig6_sta_lta.png` |
| §5.6 block-DCT baseline | `python dct_mvp.py` | `dct_mvp_results.csv` |

Default parameters match the paper: block size 250×500, high-energy fraction 0.2,
percentile classification, noise thresholding on, relative tolerances
r ∈ {0.0001, …, 0.005} as reported per table.

## 6. Notes

- The Adaptive2D method is a pure-Python wrapper: it calls the DASPack engine
  per block with the appropriate quantization step and repacks the bitstreams.
  No modification of the DASPack core is required or made.
- All compressors are evaluated under the same maximum-error bound ε
  (`max |error| ≤ ε` verified per run; see the `max_abs_err` columns).

## 7. Licence

The Adaptive2D wrapper and the experiment scripts in this archive are released
under the MIT licence (© Xinda Zhang and Hang Gao, 2026).
`code/daspack_src/` is third-party code under its own licence (see §4).

## 8. Citation

If you use this code, please cite both the paper and this release:

```bibtex
@misc{adaptive2d_code_2026,
  author  = {Zhang, Xinda and Gao, Hang},
  title   = {Adaptive2D: replication package for spatiotemporal energy-adaptive
             quantization of DAS data},
  year    = {2026},
  version = {v1.2.0},
  url     = {https://github.com/GF198910/Adaptive2D}
}
```

A citable DOI will be assigned to this release upon acceptance of the manuscript.
