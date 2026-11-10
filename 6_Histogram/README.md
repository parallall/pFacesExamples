# 6_Histogram (`histogram`)

256-bin parallel histogram using work-group `__local` memory privatization and two-level atomic accumulation.

## What This Example Demonstrates

- **Shared-memory privatization**: Each work-group accumulates counts into a fast `__local uint local_bins[256]` array via `atomic_inc`, avoiding global atomic contention.
- **Two-kernel OpenCL pipeline**: `zeroBins` clears global bins on device before `histogramCompute` merges work-group histograms into global memory via `atomic_add`.
- **Exact ground-truth verification**: Recomputes the 256-bin CPU histogram in `registerPostExecuteFunction` and verifies exact bin-by-bin integer equality and total element count (`mismatched_bins == 0`).

## Configuration Keys (`example.cfg`)

- `num_elements`: Number of 8-bit input samples >= 1 (default `262144`).
- `wg_size`: Work-group size, power of 2 in `[1, 256]` (default `64`).
- `num_groups`: Number of grid-striding work-groups >= 1 (default `64`).

## Build & Run

```sh
sh build.sh
sh test.sh
```
