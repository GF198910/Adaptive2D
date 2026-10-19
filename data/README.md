# Data

The 9 HDF5 datasets used in the paper are **not** redistributed with this archive.
Place the following files in this directory before running any script:

| Expected file name | Dataset | Shape (channels × samples) | Source |
|---|---|---|---|
| `7M_S_20231005-214004_RAW.h5` | Alboran | 5984 × 6000 | DASPack release: https://github.com/asleix/daspack |
| `ZI_C_20230609-102630_RAW.h5` | Castor | 3488 × 12000 | DASPack release |
| `9G_P.h5` | Rhonegletscher | 2176 × 6000 | DASPack release |
| `athens.h5` | Athens | 1500 × 6000 | DASPack release |
| `event_0322_a.h5` | MiDAS-0322A | 704 × 6000 | Taiwan Academia Sinica MiDAS project (upon reasonable request) |
| `event_0322_b.h5` | MiDAS-0322B | 704 × 6000 | MiDAS project |
| `event_0620_a.h5` | MiDAS-0620A | 356 × 6000 | MiDAS project |
| `event_0620_b.h5` | MiDAS-0620B | 356 × 6000 | MiDAS project |
| `event_0620_c.h5` | MiDAS-0620C | 286 × 6000 | MiDAS project |

Format: each file contains a single HDF5 dataset named `data`,
shape `(n_channels, n_samples)`, dtype float32.

Note on DASPack file names: the upstream DASPack release uses its own naming
scheme (the IDs shown in the file names above, e.g. `9G_P`); if you download the
raw upstream files, rename them to the names in the first column. The MiDAS files
were converted from the project's native format with `code/prepare_datasets.py`.
