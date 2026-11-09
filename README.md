# pFacesExamples

[![CI](https://github.com/mkhaled87/pFacesExamples/actions/workflows/ci.yml/badge.svg)](https://github.com/mkhaled87/pFacesExamples/actions/workflows/ci.yml)
[![pFaces](https://img.shields.io/badge/pFaces->=1.5-blue.svg)](https://github.com/mkhaled87/pFaces)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A collection of introductory examples, driver templates, and hardware sanity tests demonstrating how to develop custom kernel drivers and parallel programs using **pFaces**.

---

## Contents

- **`0_HelloWorld`**: A complete minimal working example demonstrating:
  - Defining an OpenCL C kernel with pFaces markup (`example.cl`).
  - Implementing a host-side C++ kernel driver (`example.cpp`).
  - Passing configuration items and kernel parameters.
  - Compiling and executing on target compute devices (CPU, GPU, or Hardware Accelerators).

---

## Prerequisites

- **pFaces**: Version `1.5` or higher installed (see [pFaces Releases](https://github.com/mkhaled87/pFaces/releases)).
- **CMake**: Version `3.22` or newer.
- **C++ Compiler**: Supporting C++14 (GCC 11+, Clang, or MSVC).
- **OpenCL Runtime**: Any OpenCL 1.2, 2.0, or 3.0 runtime (e.g., PoCL, Apple Metal, NVIDIA CUDA, Intel OpenCL).

---

## Quick Start

### 1. Build the Driver

```bash
# Build all example drivers
sh build.sh

# Or build a specific example:
cd 0_HelloWorld
sh build.sh
```

### 2. Run the Verification Tests

```bash
# Run all examples
sh test.sh

# Or run 0_HelloWorld directly:
cd 0_HelloWorld
sh test.sh
```

---

## Docker Quickstart

A pre-configured Dockerfile based on `Ubuntu 24.04` and `pFaces 1.5` is provided:

```bash
# Build the container image
docker build -t pfaces-examples .

# Run the container
docker run --rm pfaces-examples
```

---

## License

This repository is distributed under the [MIT License](LICENSE).
