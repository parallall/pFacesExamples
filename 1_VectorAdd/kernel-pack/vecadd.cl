#define VEC_N @@VEC_N@@
#define VEC_ALPHA @@VEC_ALPHA@@

kernel void vecadd(
    global const float* A,
    global const float* B,
    global float* C) {
    const size_t gid = get_global_id(0);
    if (gid < (size_t)VEC_N) {
        C[gid] = ((float)(VEC_ALPHA)) * A[gid] + B[gid];
    }
}
