# 2_MatrixMul (Tiled Shared-Memory Matrix Multiplication: C = A * B)

Demonstrates 2D `NDRange` execution, workgroup `__local` memory tiling with `barrier(CLK_LOCAL_MEM_FENCE)`, dynamic `.mem` sizing in C++, `pfacesKernelSource::updateParameters` JIT specialization (`@@MAT_M@@`, `@@MAT_K@@`, `@@MAT_N@@`, `@@TILE_SIZE@@`), `configureTuneParallelProgram` (`-t`), `-hm`, and exact triple-loop CPU ground-truth verification.

```bash
sh build.sh
sh test.sh
```
