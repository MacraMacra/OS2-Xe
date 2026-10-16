# OS2-Xe

FP64 matrix multiplication (DGEMM and ZGEMM) emulated on the INT8 matrix engines
(XMX) of Intel GPUs with **Ozaki Scheme II**. Header-only, SYCL + oneMKL,
BLAS-like interface, tunable accuracy up to correctly rounded results.

```
C = alpha * op(A) * op(B) + beta * C      column-major, op = N | T | C
```

On an Intel Arc Pro B60 (Xe2 / Battlemage, 24 GB) the emulated DGEMM runs
**12–28× faster** than the native FP64 oneMKL DGEMM of the same board, while
matching or exceeding its accuracy.

| N = 8192, DGEMM | s = 8 | s = 12 | s = 16 | native oneMKL DGEMM |
|---|---:|---:|---:|---:|
| TFLOP/s | 17.3 | 12.1 | 9.3 | 0.76 |

## Features

- `dgemm` and `zgemm` with the full BLAS parameter set: `op(A)`, `op(B)` (N, T, C),
  `alpha`, `beta`, leading dimensions, arbitrary `m, n, k`.
- Accuracy is controlled by the number of moduli `s` (2…24): `s = 14–15` gives
  DGEMM-level accuracy, `s = 16` gives correctly rounded results for well-scaled
  data, `s = 20–24` handles inputs with a wide exponent range.
- Fully device-resident: `A`, `B`, `C` are device USM pointers, no host round
  trips, no internal synchronisation — every call returns a `sycl::event`.
- Deterministic: all exact arithmetic is integer, results are bit-identical
  between runs and for any tiling.
- ZGEMM uses the 3M (Karatsuba) form evaluated exactly in integer arithmetic:
  `3s` INT8 GEMMs instead of `4s`.
- Automatic tiling to fit into the available VRAM, k-splitting for `k > 131071`.

## Requirements

- Intel oneAPI Base Toolkit (DPC++/C++ Compiler `icpx` and oneMKL), tested with 2026.1
- An Intel GPU with XMX engines; developed and measured on Intel Arc Pro B60
  (`bmg-g21`). Other Xe2-based devices and newer should work with a JIT build.
- Linux with the Level Zero runtime

## Quick start

```bash
git clone https://github.com/MacraMacra/OS2-Xe.git
```

The library is two headers in [`src/`](src/). Add `src/` to the include path:

```cpp
#include "ozaki2_gpu.hpp"

sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
oz2::Ozaki2 oz(q);                                  // keeps the workspace between calls

int s = oz2::Ozaki2::moduli_for_bits(k, 53);        // or any s in [2, 24]
oz.gemm(oz2::Trans::N, oz2::Trans::N, m, n, k,
        1.0, dA, lda, dB, ldb, 0.0, dC, ldc, s).wait();

oz.zgemm(oz2::Trans::C, oz2::Trans::N, m, n, k,     // std::complex<double>*
         alpha, zA, lda, zB, ldb, beta, zC, ldc, s).wait();
```

Compile with:

```bash
source /opt/intel/oneapi/setvars.sh
icpx -fsycl -fsycl-device-code-split=per_kernel -O3 -IOS2-Xe/src app.cpp -qmkl
```

Add `-fsycl-targets=spir64_gen -Xs "-device bmg-g21"` for an ahead-of-time build
for Arc B-series. With a JIT build keep `-fsycl-device-code-split=per_kernel`:
otherwise the first call compiles all ~150 kernel instantiations at once.

## API

Namespace `oz2`, class `Ozaki2`:

| member | description |
|---|---|
| `Ozaki2(sycl::queue& q)` | binds to a queue (must outlive the object); in-order queue recommended |
| `gemm(ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, s, deps = {})` | real GEMM, `double*` device pointers |
| `zgemm(ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, s, deps = {})` | complex GEMM, `std::complex<double>*` device pointers |
| `static precision_bits(k, s)` | bits kept per element of `op(A)` / `op(B)` after scaling |
| `static precision_bits_complex(k, s)` | the same for `zgemm` |
| `static moduli_for_bits(k, bits)` | smallest `s` with `precision_bits(k, s) >= bits` |
| `set_workspace_limit(bytes)` | cap the workspace size (0 = automatic) |
| `set_resident_hint(bytes)` | VRAM held by the caller, subtracted from the workspace budget |
| `release_workspace()` | free the workspace now |
| `stage_times()` | per-phase timing of the last call (queue with `enable_profiling`) |
| `last_tiling()` | tile sizes and workspace of the last call |

`Trans::N`, `Trans::T`, `Trans::C` (conjugate transpose; equals `T` for `gemm`).

### Choosing the number of moduli

| goal | s |
|---|---|
| DGEMM-level accuracy | 14–15 |
| correctly rounded, well-scaled data (e.g. U[-1, 1)) | 15–16 |
| correctly rounded, wide exponent range within a row/column | 19–24 |

Each row of `op(A)` and each column of `op(B)` is scaled to integers with
`tk = precision_bits(k, s)` significant bits (`tk ≈ (log2 M − 2 − log2 k) / 2`,
where `M` is the product of the moduli). Speed scales roughly as `1/s`.

### Memory

The workspace is `s·(mk + kn + 4mn)` bytes per tile (plus `8·k·nb` for
`op(B) = Bᵀ`). If it does not fit, `C` is split into tiles. The budget is taken
from the driver's free-memory report; since the `xe` driver allows VRAM
oversubscription, `set_resident_hint()` or `set_workspace_limit()` keeps the
workspace inside physical VRAM.

## Build, test, examples

```bash
source /opt/intel/oneapi/setvars.sh
make -j            # AOT for bmg-g21 (needs ocloc); binaries go to build/
make -j AOT=0      # JIT build for any Intel GPU
make check         # correctness tests
make check-quick   # without the large cases
```

| directory | contents |
|---|---|
| [`src/`](src/) | the library: `ozaki2_gpu.hpp`, `coprime_tables.hpp` |
| [`tests/`](tests/) | `test_ozaki2` — element-wise rigorous error bound against a double-double reference, bit-exact integer inputs, worst case `\|C'\| = M/4`, all transpositions, `alpha`/`beta`, `ldc > m`, tiling, `k > 131071`, Inf/NaN, unit tests of the split and CRT kernels |
| [`examples/`](examples/) | `dgemm` — basic real GEMM vs. oneMKL; `zgemm` — conjugate-transposed complex GEMM; `profiling` — speed/accuracy trade-off with per-phase timing |
| [`benchmarks/`](benchmarks/) | `bench_ozaki2` — throughput; `bench_accuracy` — ulp error and fraction of correctly rounded entries, with an Ozaki Scheme I baseline |

```bash
./build/dgemm 4096                    # m [n k]
./build/zgemm 2048 2048 2048 16       # m n k s
./build/profiling 8192                # N

./build/bench_ozaki2 --sizes 4096,8192,4000x8000x4000 --s 8,12,16 --reps 5 --dgemm --check
./build/bench_accuracy --k 1024,4096 --s 4-24 --dist urand,phi2 [--complex] [--cpu]
```

`bench_ozaki2` options: `--sizes` (`N` or `MxKxN`), `--s`, `--reps`, `--dgemm`
(native oneMKL GPU baseline), `--cpu` (MKL CPU baseline), `--check`,
`--dist sincos|phi<x>`. `bench_accuracy` prints CSV: `ulp_max`, `ulp_mean`,
`cr_frac` (fraction of correctly rounded entries), `ab_max`, `fro_rel`.

## How it works

1. **Scaling.** `max|·|` over rows of `op(A)` and columns of `op(B)` gives power-of-two
   exponents `D_i`, `E_j`.
2. **Scale + split in one pass.** `x = rint(a·2^D_i)` is extracted from the bits of the
   double with round-to-nearest-even (FP64 is 16× slower than FP32 on Xe2), and all `s`
   symmetric INT8 residues are computed with 21-bit limbs and Barrett reduction, branch-free.
3. **`s` INT8 GEMMs** in one oneMKL `gemm_batch` call.
4. **Division-free CRT.** The quotient is computed in FP32 with a proven margin
   (`|C'| ≤ M/4`), the exact integer is reconstructed in 16-bit limbs with 32-bit
   accumulators, rounded once to double and scaled back exactly.
5. Kernels are instantiated per `s` with fully unrolled loops; `Bᵀ` is transposed
   through SLM instead of using the slow `transb = trans` INT8 path; tiling avoids
   the performance cliffs of the INT8 GEMM on Xe2.

Details, proofs and the full evaluation are in the paper (see [Citation](#citation)).

## Results (Intel Arc Pro B60, oneAPI 2026.1)

- DGEMM: 17.3 / 12.1 / 9.3 TFLOP/s at `s = 8 / 12 / 16`, `N = 8192`; up to 21.1 TFLOP/s
  (`N = 20480`, `s = 8`). Native oneMKL DGEMM: 0.76 TFLOP/s.
- ZGEMM: up to 27.4 TFLOP/s, 13.7–17.8× faster than native ZGEMM for `N ≥ 4096` at `s = 16`.
- 75–90% of the run time is INT8 GEMM at ≈95% of its attainable rate.
- With `s ≥ 16` results are correctly rounded for well-scaled inputs; native DGEMM
  rounds only 2–13% of the entries correctly.
- For `N ≤ 256` the kernel-launch overhead dominates and native DGEMM is faster.

## Limitations

- `s ∈ [2, 24]`.
- The scale is shared per row of `op(A)` / column of `op(B)`: if the exponent range
  within a row exceeds `tk − 53` bits, the smallest elements lose bits (increase `s`).
- Inf/NaN in a row of `op(A)` or column of `op(B)` turns the whole corresponding row/column
  of the result into NaN (not element-wise IEEE semantics).
- Results in the subnormal range may be double-rounded.

## Citation

If you use this library, please cite:

```bibtex
@misc{rostovskis2026ozaki2xe2,
  title  = {Emulating {DGEMM} and {ZGEMM} with {INT8} Matrix Engines on {Intel} {Arc} {GPUs}:
            {Ozaki} Scheme {II} on the {Xe2} Architecture},
  author = {Rostovskis, Rostislavs and Platonenko, Aleksandrs},
  year   = {2026},
  note   = {Institute of Solid State Physics, University of Latvia},
  url    = {https://github.com/MacraMacra/OS2-Xe}
}
```

The method itself: K. Ozaki, Y. Uchino, T. Imamura, *Ozaki Scheme II: A GEMM-oriented
emulation of floating-point matrix multiplication using an integer modular technique*,
[arXiv:2504.08009](https://arxiv.org/abs/2504.08009) (2025).

## License

[MIT](LICENSE)
