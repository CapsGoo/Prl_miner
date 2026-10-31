// ghost_kernel v0: int8 tensor-core GEMM tile engine (Pearl PoUW compute core prototype)
// C(int32) = A(int8) x B(int8), tile 4096, benchmark mma throughput on sm_86
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <mma.h>
#include <cuda_runtime.h>
#include <time.h>

using namespace nvcuda;

#define BM 128
#define BN 64
#define BK 32
#define KDIM 4096

__global__ void gemm_int8(const int8_t *A, const int8_t *B, int32_t *C, int M, int N) {
    __shared__ int8_t sA[BM][BK];
    __shared__ int8_t sB[BK][BN];

    int wm = threadIdx.y / 2;         // (32,8): warp = rows 2y,2y+1 -> wm 0..3
    int lane = threadIdx.x % 32;

    int warpM = blockIdx.y * BM + wm * 32;
    int warpN = blockIdx.x * BN;            // warp covers full 64-wide tile

    wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc[2][4];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 4; j++)
            wmma::fill_fragment(acc[i][j], 0);

    for (int k0 = 0; k0 < KDIM; k0 += BK) {
        // load tiles to smem
        for (int idx = threadIdx.y * 32 + threadIdx.x; idx < BM * BK; idx += 256) {
            int r = idx / BK, c = idx % BK;
            int gr = blockIdx.y * BM + r;
            int gc = k0 + c;
            sA[r][c] = A[(long)gr * KDIM + gc];
        }
        for (int idx = threadIdx.y * 32 + threadIdx.x; idx < BK * BN; idx += 256) {
            int r = idx / BN, c = idx % BN;
            int gr = k0 + r;
            int gc = blockIdx.x * BN + c;
            sB[r][c] = B[(long)gr * N + gc];
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> fa[2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::row_major> fb[4];
        for (int kk = 0; kk < BK; kk += 16) {
            for (int i = 0; i < 2; i++)
                wmma::load_matrix_sync(fa[i], &sA[i * 16][kk], BK);
            for (int j = 0; j < 4; j++)
                wmma::load_matrix_sync(fb[j], &sB[kk][j * 16], BN);
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 4; j++)
                    wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
        __syncthreads();
    }

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 4; j++)
            wmma::store_matrix_sync(&C[(long)(warpM + i * 16) * N + warpN + j * 16], acc[i][j], N, wmma::mem_row_major);
}

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    int M = 4096, N = 4096;
    int iters = argc > 1 ? atoi(argv[1]) : 50;
    int8_t *A, *B; int32_t *C;
    cudaMallocManaged(&A, (size_t)M * KDIM);
    cudaMallocManaged(&B, (size_t)KDIM * N);
    cudaMallocManaged(&C, (size_t)M * N * 4);
    for (size_t i = 0; i < (size_t)M * KDIM; i++) A[i] = (int8_t)(rand() % 127);
    for (size_t i = 0; i < (size_t)KDIM * N; i++) B[i] = (int8_t)(rand() % 127);

    dim3 block(32, 8);
    dim3 grid(N / BN, M / BM);
    gemm_int8<<<grid, block>>>(A, B, C, M, N);
    cudaDeviceSynchronize();

    double t0 = now();
    for (int it = 0; it < iters; it++) gemm_int8<<<grid, block>>>(A, B, C, M, N);
    cudaDeviceSynchronize();
    double t = now() - t0;
    double tops = 2.0 * M * N * KDIM * iters / t / 1e12;
    printf("ghost wmma int8: %dx%dx%d  %.2f TOPS (%.2f ms/iter)\n", M, N, KDIM, tops, t / iters * 1000);
    printf("C[1234]=%d\n", C[1234]);
    return 0;
}
