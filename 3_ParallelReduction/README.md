# 3_ParallelReduction (Two-Stage Tree Sum Reduction)

Demonstrates multi-kernel chaining (`reducePartial` -> `reduceFinal`), OpenCL workgroup shared-memory (`local float* scratch`) tree reductions via `.mem` (`isLocalMemory = "true"`), resident buffer sharing (`residentArgRef`), non-power-of-two array sizes via strided stage-1 reduction, and compensated double-precision CPU ground-truth verification.

```bash
sh build.sh
sh test.sh
```
