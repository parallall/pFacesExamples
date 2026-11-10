#define MAT_M @@MAT_M@@
#define MAT_K @@MAT_K@@
#define MAT_N @@MAT_N@@
#define TILE_SIZE @@TILE_SIZE@@

kernel void matmul(
    global const float* A,
    global const float* B,
    global float* C) {
    local float tileA[TILE_SIZE][TILE_SIZE];
    local float tileB[TILE_SIZE][TILE_SIZE];

    const size_t col = get_global_id(0);
    const size_t row = get_global_id(1);
    const size_t lx = get_local_id(0);
    const size_t ly = get_local_id(1);

    float acc = 0.0f;
    const size_t numTiles = ((size_t)MAT_K + (size_t)TILE_SIZE - 1UL) / (size_t)TILE_SIZE;
    for (size_t t = 0; t < numTiles; t++) {
        const size_t tiledCol = t * (size_t)TILE_SIZE + lx;
        const size_t tiledRow = t * (size_t)TILE_SIZE + ly;

        tileA[ly][lx] = (row < (size_t)MAT_M && tiledCol < (size_t)MAT_K)
            ? A[row * (size_t)MAT_K + tiledCol]
            : 0.0f;
        tileB[ly][lx] = (tiledRow < (size_t)MAT_K && col < (size_t)MAT_N)
            ? B[tiledRow * (size_t)MAT_N + col]
            : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (size_t k = 0; k < (size_t)TILE_SIZE; k++) {
            acc += tileA[ly][k] * tileB[k][lx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < (size_t)MAT_M && col < (size_t)MAT_N) {
        C[row * (size_t)MAT_N + col] = acc;
    }
}
