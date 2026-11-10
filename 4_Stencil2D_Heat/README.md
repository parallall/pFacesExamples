# 4_Stencil2D_Heat (`heat2d`)

2D 5-point Jacobi heat-diffusion stencil solved on a regular `width x height` grid with Dirichlet boundary conditions (`T = 100.0` on top edge `y = 0`, `T = 0.0` on all other boundaries and initial interior).

## What This Example Demonstrates

- **2D ND-Range execution**: Dispatches two-dimensional work-groups `(wg_x, wg_y)` over a padded global grid.
- **Resident ping-pong device buffers**: Keeps `grid_a` and `grid_b` on device across iterations via `isNewOrResident = "false"` and `residentArgRef`.
- **pFaces VM host-controlled iteration loop**: Uses `setAsBlockingSyncPoint()`, `setAsHostFunction(...)`, and `setAsJumpNe(...)` to drive `iterations / 2` A->B->A Jacobi pairs without host transfers between steps.
- **Ground-truth verification**: Re-runs the exact double-buffered CPU Jacobi relaxation in a `registerPostExecuteFunction` callback and fails with a non-zero exit code if `max_abs_err > 1e-4`.

## Configuration Keys (`example.cfg`)

- `width`: Grid width (must be >= 3; default `128`).
- `height`: Grid height (must be >= 3; default `128`).
- `iterations`: Even number of Jacobi iterations >= 2 (default `100`).
- `diffusion`: Stencil weight `r` in `(0, 0.25]` (default `0.2`).
- `wg_x`, `wg_y`: 2D local work-group dimensions (default `16 x 16`).

## Build & Run

```sh
sh build.sh
sh test.sh
```
