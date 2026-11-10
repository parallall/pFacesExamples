// Shared tiled all-pairs gravitational force + symplectic Euler update.
inline void nbodyStepImpl(
    __global float4* pos_in,
    __global float4* pos_out,
    __global float4* vel,
    __local float4* tile_pos) {
  const uint gid = (uint)get_global_id(0);
  const uint lid = (uint)get_local_id(0);
  const uint wgSize = (uint)get_local_size(0);
  const uint n = (uint)(@@NBODY_COUNT@@);
  const uint numTiles = (n + wgSize - 1U) / wgSize;

  const float dt = (float)(@@NBODY_DT@@);
  const float eps2 = (float)(@@NBODY_EPS2@@);

  const bool active = (gid < n);
  const float4 myPos = active ? pos_in[gid] : (float4)(0.0f, 0.0f, 0.0f, 0.0f);
  float3 acc = (float3)(0.0f, 0.0f, 0.0f);

  // Every work-item participates in tile loads and barriers to prevent divergence deadlocks.
  for (uint tile = 0U; tile < numTiles; tile++) {
    const uint j = tile * wgSize + lid;
    tile_pos[lid] = (j < n) ? pos_in[j] : (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (active) {
      for (uint k = 0U; k < wgSize; k++) {
        const float4 other = tile_pos[k];
        const float3 r = other.xyz - myPos.xyz;
        const float distSqr = dot(r, r) + eps2;
        const float invDist = 1.0f / sqrt(distSqr);
        const float invDistCube = invDist * invDist * invDist;
        acc += (other.w * invDistCube) * r;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (active) {
    float4 myVel = vel[gid];
    myVel.xyz += acc * dt;
    const float4 nextPos = (float4)(myPos.xyz + myVel.xyz * dt, myPos.w);
    vel[gid] = myVel;
    pos_out[gid] = nextPos;
  }
}

__kernel void nbodyStepAB(
    __global float4* pos_in,
    __global float4* pos_out,
    __global float4* vel,
    __local float4* tile_pos) {
  nbodyStepImpl(pos_in, pos_out, vel, tile_pos);
}

__kernel void nbodyStepBA(
    __global float4* pos_in,
    __global float4* pos_out,
    __global float4* vel,
    __local float4* tile_pos) {
  nbodyStepImpl(pos_in, pos_out, vel, tile_pos);
}
