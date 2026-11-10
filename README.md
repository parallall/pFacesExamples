# pFacesExamples

[![CI](https://github.com/parallall/pFacesExamples/actions/workflows/ci.yml/badge.svg)](https://github.com/parallall/pFacesExamples/actions/workflows/ci.yml)
[![pFaces](https://img.shields.io/badge/pFaces->=1.5-blue.svg)](https://github.com/parallall/pFaces)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Canonical data-parallel OpenCL and C++ driver examples for **pFaces** (matching classic OpenCL/CUDA patterns), complete with build scripts, profiling tests, host-pinned (`-hm`) execution, and automated CPU ground-truth verification.

---

## Contents

| Example | Kernel | Core Parallel Pattern & pFaces Features Demonstrated | Ground-Truth Verification |
| :--- | :--- | :--- | :--- |
| **`0_HelloWorld`** | `example` | Minimal kernel driver, config/parameter macros (`config.template`, `@@param@@`), device-to-host buffer readback | Verifies `"pFcs"` handshake token (`[PASS]`) |
| **`1_VectorAdd`** | `vecadd` | Element-wise 1D SAXPY (`C = alpha * A + B`), `isRangeScalable = "true"` | Compares every element against host FP32 reference (`max_abs_err <= 1e-5`) |
| **`2_MatrixMul`** | `matmul` | 2D ND-Range, `__local` memory tiling + `barrier(CLK_LOCAL_MEM_FENCE)`, auto-tuner `-t` (`configureTuneParallelProgram`), dynamic `.mem` sizing | Compares against exact `(M x K) * (K x N)` CPU reference (`max_abs_err <= 1e-3`) |
| **`3_ParallelReduction`** | `reduction` | Two-stage tree reduction in `__local` memory, resident buffers via `residentArgRef`, multi-kernel sequencing | Compares against double-precision Kahan summation (`rel_err <= 1e-4`) |
| **`4_Stencil2D_Heat`** | `heat2d` | 2D 5-point Jacobi heat diffusion, ping-pong device buffers, VM iteration loop via `setAsHostFunction` + `setAsJumpNe` | Compares converged field against double-buffered CPU reference (`max_abs_err <= 1e-4`) |
| **`5_NBody`** | `nbody` | Tiled all-pairs 3D gravitational N-body with `float4` vectors and `__local` position caching inside a VM ping-pong loop | Compares all positions & velocities against `O(N^2)` CPU simulation |
| **`6_Histogram`** | `histogram` | 256-bin histogram using `__local uint[256]` privatization (`atomic_inc`) and global `atomic_add` merge | Exact integer equality across all 256 bins (`mismatched_bins == 0`) |

---

## Prerequisites

- **pFaces**: Version `1.5` or higher installed (see [pFaces Releases](https://github.com/parallall/pFaces/releases)).
- **CMake**: Version `3.22` or newer.
- **C++ Compiler**: Supporting C++14 or newer (GCC 11+, Clang, or MSVC).
- **OpenCL Runtime**: Any OpenCL 1.2, 2.0, or 3.0 runtime (e.g., PoCL, Apple Metal, NVIDIA CUDA, Intel OpenCL).

---

## Quick Start

### 1. Build All Drivers

```bash
# Build all example drivers (0_HelloWorld .. 6_Histogram)
sh build.sh

# Or build a specific example:
cd 2_MatrixMul
sh build.sh
```

### 2. Run All Verification Tests

Each example verifies its OpenCL device output against host ground truth inside `registerPostExecuteFunction` across default config, custom `.cfg` + profiling (`-p`), and host-pinned shared memory (`-hm`), returning a non-zero exit code on any mismatch.

```bash
# Build (if needed) and run all 7 examples
sh test.sh

# Or run a specific example directly:
cd 4_Stencil2D_Heat
sh test.sh
```

---

## Docker Quickstart

A pre-configured `Dockerfile` based on `Ubuntu 24.04`, PoCL CPU OpenCL, and `pFaces 1.5` is provided:

```bash
# Build the container image
docker build -t pfaces-examples .

# Run all examples inside the container
docker run --rm pfaces-examples
```

---

## License

This repository is distributed under the [MIT License](LICENSE).
