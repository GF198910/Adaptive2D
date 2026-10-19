<p align="center">
  <img src="docs/assets/logo.svg" alt="DASPack Logo" />
</p>

# DASPack: Controlled data compression for Distributed Acoustic Sensing

DASPack is a fast, open-source compressor for huge Distributed Acoustic Sensing (DAS) datasets.  
It supports **lossless** and **fixed-accuracy lossy** modes, letting you store data with an exact bound on reconstruction error. Read the [paper](https://arxiv.org/abs/2507.16390) to find out more!

The core is written in Rust for speed and safety, with a thin Python API for convenient integration into your workflows.

We’re continuously improving the code to serve everyone better. Feel free to share your use cases or report any issues in the [Discussions section](https://github.com/asleix/daspack/discussions)!

---

## ✨ Highlights

- **Lossless or fixed-accuracy** — pick zero error or a max absolute error and get exactly what you asked for.
- **Multi-threaded** — control the number of threads per encode/decode call.
- **High throughput** — 800 MB/s+ on an 8-core laptop in typical workloads.
- **Pure Rust core** — no unsafe C buffers exposed to user code.
- **Python bindings** — direct `encode` / `decode` interface for NumPy arrays.

---

## 🚀 Quick start

### 1. Install (Python ≥ 3.9)

```bash
pip install daspack-dev
# or, from source (Rust ≥ 1.74):
# maturin develop --release
```

### 2. Encode and store with h5py

You can store the compressed DASPack bitstream as raw bytes in HDF5:

```python
import numpy as np, h5py
from daspack import DASCoder, Quantizer

# Example: lossless compression with 4 threads
data = np.random.randint(-1000, 1000, size=(4096, 8192), dtype=np.int32)
coder = DASCoder(threads=4)

# Encode in Lossless mode
stream = coder.encode(
    data,
    Quantizer.Lossless(),
)

with h5py.File("example.h5", "w") as f:
    f.create_dataset("compressed", data=np.frombuffer(stream, dtype=np.uint8))
```

### 3. Read and decode

```python
import numpy as np, h5py
from daspack import DASCoder

coder = DASCoder(threads=4)

with h5py.File("example.h5") as f:
    raw = f["compressed"][:].tobytes()

# Decode: dtype is inferred from the stream
restored = coder.decode(raw)
```


### 4. Lossy example with fixed error bound

```python
import numpy as np
from daspack import DASCoder, Quantizer

# Generate some example data
data = np.random.uniform(-100, 100, size=(6, 8)).astype(np.float64)

coder = DASCoder(threads=2)

# Target: absolute error ≤ step/2
step = 0.5

# Encode with Uniform quantizer (lossy) and given step
stream = coder.encode(
    data,
    Quantizer.Uniform(step=step),
)

# Decode (dtype inferred from stream)
restored = coder.decode(stream)

# Verify bound
tol = step / 2 + 1e-12
max_err = np.max(np.abs(restored - data))
print(f"Max abs error: {max_err:.6f} (tolerance {tol})")
assert max_err <= tol

print("Original data:\n", data)
print("Restored data:\n", restored)
```

The expected output is
```
Max abs error: 0.250000 (tolerance 0.250000)
Original data:
 [[ ... ]]
Restored data:
 [[ ... ]]
```

---


## ⚙️ How it works

```
(float mode) Quantize → Wavelet (5/3) → 2-D LPC → Arithmetic coding
(int mode)   Identity  → Wavelet (5/3) → 2-D LPC → Arithmetic coding
```
The lossy path is bounded-error thanks to uniform quantization; the rest of the chain is perfectly reversible.

Read the paper (see citation below!) for more information 😄

### Parameters at a glance

- **threads** *(int, default=1)* — number of threads for compression/decompression.
- **quantizer** *(Quantizer)* — choose `Lossless()` or `Uniform(step=...)`.
- **blocksize** *((int, int), default=(1000, 1000))* — 2D block size used for compression.
- **levels** *(int, default=1)* — predictor levels across dimensions.
- **order** *(int, default=1)* — prediction order.

---

## 📄 License

DASPack is released under the 3-Clause BSD License.

---

## 🤝 Contributing

Bug reports and pull requests are welcome.
If you plan a large change, please open an issue first so we can discuss the design.

---

## 📣 Citing

If you use DASPack in academic work, please cite:

> Seguí, A., Ugalde, A., Fichtner, A., Ventosa, S., & Morros, J. R. (2025). **DASPack: Controlled Data Compression for Distributed Acoustic Sensing**. *Geophysical Journal International*.\
> DOI: 10.1093/gji/ggaf397

Thanks for supporting open science!

