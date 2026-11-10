#define HIST_BINS 256U

__kernel void zeroBins(__global uint* bins) {
  const uint gid = (uint)get_global_id(0);
  if (gid < HIST_BINS) {
    bins[gid] = 0U;
  }
}

__kernel void histogramCompute(
    __global uint* in_data,
    __global uint* bins,
    __local uint* local_bins) {
  const uint lid = (uint)get_local_id(0);
  const uint wgSize = (uint)get_local_size(0);
  const uint gid = (uint)get_global_id(0);
  const uint totalThreads = (uint)get_global_size(0);
  const uint n = (uint)(@@HIST_N@@);

  for (uint b = lid; b < HIST_BINS; b += wgSize) {
    local_bins[b] = 0U;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint i = gid; i < n; i += totalThreads) {
    const uint bin = in_data[i] & 0xFFU;
    atomic_inc(&local_bins[bin]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint b = lid; b < HIST_BINS; b += wgSize) {
    const uint count = local_bins[b];
    if (count != 0U) {
      atomic_add(&bins[b], count);
    }
  }
}
