#define RED_N @@RED_N@@
#define NUM_GROUPS @@NUM_GROUPS@@

kernel void reducePartial(
    global const float* in_data,
    global float* partial_sums,
    local float* scratch) {
    const size_t gid = get_global_id(0);
    const size_t gsize = get_global_size(0);
    const size_t lid = get_local_id(0);
    const size_t lsize = get_local_size(0);
    const size_t grp = get_group_id(0);

    float localAcc = 0.0f;
    for (size_t idx = gid; idx < (size_t)RED_N; idx += gsize) {
        localAcc += in_data[idx];
    }
    scratch[lid] = localAcc;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (size_t stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        partial_sums[grp] = scratch[0];
    }
}

kernel void reduceFinal(
    global const float* partial_sums,
    global float* final_sum,
    local float* scratch) {
    const size_t lid = get_local_id(0);
    const size_t lsize = get_local_size(0);

    scratch[lid] = (lid < (size_t)NUM_GROUPS) ? partial_sums[lid] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (size_t stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        final_sum[0] = scratch[0];
    }
}
