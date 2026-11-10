# 5_NBody (`nbody`)

Tiled all-pairs 3D gravitational N-body simulation with Plummer softening and symplectic Euler integration.

## What This Example Demonstrates

- **Shared-memory force tiling**: Caches chunks of particle positions+masses (`float4`) into `__local` memory with `barrier(CLK_LOCAL_MEM_FENCE)` to reduce global memory traffic from `O(N^2)` to `O(N^2 / wg_size)`.
- **Vectorized `float4` global buffers**: Stores particle positions+masses and velocities using 16-byte aligned `cl_float4` elements (`baseTypeSize = "16"`).
- **VM ping-pong time-stepping**: Alternates `nbodyStepAB` and `nbodyStepBA` across `steps / 2` pairs via `setAsHostFunction(...)` + `setAsJumpNe(...)`.
- **Ground-truth verification**: Compares all final particle positions and velocities against an exact `O(N^2)` CPU simulation inside `registerPostExecuteFunction` and fails with a non-zero exit code on mismatch.

## Configuration Keys (`example.cfg`)

- `num_bodies`: Number of simulated bodies >= 1 (default `1024`).
- `steps`: Even number of integration steps >= 2 (default `10`).
- `dt`: Integration time step > 0 (default `0.01`).
- `softening`: Plummer softening length > 0 (default `0.2`).
- `wg_size`: 1D work-group size (tile width, default `64`).

## Build & Run

```sh
sh build.sh
sh test.sh
```
