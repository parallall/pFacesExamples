# 1_VectorAdd (SAXPY: C = alpha * A + B)

Demonstrates element-wise 1D memory-bound OpenCL execution (`get_global_id(0)`), `.mem` buffer layout (`isRangeScalable`), configuration parsing (`config.defaults` / `.cfg`), device/host-memory (`-hm`) compatibility, and host-side ground-truth verification in `registerPostExecuteFunction`.

```bash
sh build.sh
sh test.sh
```
