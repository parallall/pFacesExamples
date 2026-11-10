// Shared helper implementing a 5-point 2D Jacobi heat-diffusion step.
inline void jacobiStepImpl(__global float* grid_in, __global float* grid_out) {
  const uint x = (uint)get_global_id(0);
  const uint y = (uint)get_global_id(1);
  const uint w = (uint)(@@HEAT_WIDTH@@);
  const uint h = (uint)(@@HEAT_HEIGHT@@);

  if (x >= w || y >= h) {
    return;
  }

  const uint idx = y * w + x;
  if (x == 0U || x == (w - 1U) || y == 0U || y == (h - 1U)) {
    grid_out[idx] = grid_in[idx];
    return;
  }

  const float r = (float)(@@HEAT_R@@);
  const float centerWeight = 1.0f - 4.0f * r;
  const float center = grid_in[idx];
  const float north = grid_in[idx - w];
  const float south = grid_in[idx + w];
  const float west = grid_in[idx - 1U];
  const float east = grid_in[idx + 1U];
  grid_out[idx] = centerWeight * center + r * (north + south + west + east);
}

__kernel void jacobiStepAB(__global float* grid_in, __global float* grid_out) {
  jacobiStepImpl(grid_in, grid_out);
}

__kernel void jacobiStepBA(__global float* grid_in, __global float* grid_out) {
  jacobiStepImpl(grid_in, grid_out);
}
