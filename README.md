# pFacesExamples

[![CI](https://github.com/parallall/pFacesExamples/actions/workflows/ci.yml/badge.svg)](https://github.com/parallall/pFacesExamples/actions/workflows/ci.yml)
[![pFaces](https://img.shields.io/badge/pFaces->=1.5-blue.svg)](https://github.com/parallall/pFaces)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](0_HelloWorld/LICENSE)

A progressive 7-part learning curriculum introducing canonical data-parallel algorithms (matching classic OpenCL and CUDA patterns) in **pFaces** — moving step-by-step from kernel/driver basics to shared-memory tiling, tree reductions, VM ping-pong loops, and atomic privatization, complete with host ground-truth verification.

---

## Learning Curriculum (`0` → `6`)

| Lesson | Example | Core Parallel Concept & pFaces Features Introduced | Builds On | Ground-Truth Verification |
| :--- | :--- | :--- | :--- | :--- |
| **0** | [**`0_HelloWorld`**](0_HelloWorld/README.md) | Kernel pack vs C++ driver anatomy, `@pfaces-configValueString` & `@@param@@` macros, VM messages & host readback | *Start here* | Verifies `"pFcs"` handshake token (`[PASS]`) |
| **1** | [**`1_VectorAdd`**](1_VectorAdd/README.md) | 1D element-wise SAXPY (`get_global_id(0)`), `isRangeScalable = "true"` buffer scaling, `-hm` vs discrete DMA writes | Lesson 0 | Compares every element against host FP32 reference (`max_abs_err <= 1e-5`) |
| **2** | [**`2_MatrixMul`**](2_MatrixMul/README.md) | 2D ND-Range, `__local` shared-memory tiling + `barrier(CLK_LOCAL_MEM_FENCE)`, dynamic `.mem` sizing, auto-tuner (`-t`) | Lesson 1 | Compares against exact `(M x K) * (K x N)` CPU reference |
| **3** | [**`3_ParallelReduction`**](3_ParallelReduction/README.md) | Many-to-one binary tree reduction, grid-stride loops, dynamic `isLocalMemory`, multi-kernel chaining (`residentArgRef`) | Lesson 2 | Compares against double-precision Kahan summation (`rel_err <= 1e-4`) |
| **4** | [**`4_Stencil2D_Heat`**](4_Stencil2D_Heat/README.md) | 2D 5-point Jacobi PDE stencil, ping-pong double buffering, VM loops (`setAsHostFunction` + `setAsJumpNe`) | Lessons 2, 3 | Compares converged field against double-buffered CPU reference (`max_abs_err <= 1e-4`) |
| **5** | [**`5_NBody`**](5_NBody/README.md) | All-pairs $O(N^2)$ tiled force accumulation, 16-byte `float4` vectors, divergence-safe tile barriers inside a VM ping-pong loop | Lessons 2, 4 | Compares all positions & velocities against $O(N^2)$ CPU simulation |
| **6** | [**`6_Histogram`**](6_Histogram/README.md) | Scatter-add privatization (`__local uint[256]`), `atomic_inc` / `atomic_add`, device buffer clear (`zeroBins`) | Lessons 3, 5 | Exact 256-bin integer equality (`mismatched_bins == 0`) |

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

# Or build a specific lesson:
cd 2_MatrixMul
sh build.sh
```

### 2. Run All Verification Tests

Each lesson verifies its OpenCL device output against host ground truth inside `registerPostExecuteFunction` across default config, custom `.cfg` + profiling (`-p`), and host-pinned shared memory (`-hm`), returning a non-zero exit code on any mismatch.

```bash
# Build (if needed) and run all 7 lessons
sh test.sh

# Or run a specific lesson directly:
cd 4_Stencil2D_Heat
sh test.sh
```

---

## Docker Quickstart

A pre-configured `Dockerfile` based on `Ubuntu 24.04`, PoCL CPU OpenCL, and `pFaces 1.5` is provided:

```bash
# Build the container image
docker build -t pfaces-examples .

# Run all lessons inside the container
docker run --rm pfaces-examples
```

---

## License

This repository is distributed under the [MIT License](0_HelloWorld/LICENSE).
