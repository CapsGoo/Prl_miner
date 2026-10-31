// ghost_v1: int8 mma.sync + cp.async pipeline for sm_86 (Pearl PoUW compute core)
// 128x128 block tile, BK=64, 2-stage cp.async, 8 warps (2x4, warp tile 64x32)
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#include <time.h>

#define BM 128
#define BN 128
#define BK 64
#define APITCH 80   // bytes per smem row for A (64+16 pad)
#define BPITCH 80   // B stored transposed: row = n, pitch BK+16

__device__ inline void cp16(unsigned sdst, const void *gsrc) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sdst), "l"(gsrc));
}
__device__ inline void cp_commit() { asm volatile("cp.async.commit_group;"); }
__device__ inline void cp_wait0() { asm volatile("cp.async.wait_group 0;"); }
__device__ inline void cp_wait1() { asm volatile("cp.async.wait_group 1;"); }

__device__ inline void ldm_x4(unsigned saddr, uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, bool trans) {
    if (trans)
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(saddr));
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(saddr));
}

__device__ inline void mma_s8(int32_t &d0, int32_t &d1, int32_t &d2, int32_t &d3,
                              uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                              uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

extern __shared__ int8_t smem[];  // 2 stages: A(BM*BK) + B(BK*BN) each

__global__ void gemm8(const int8_t *__restrict__ A, const int8_t *__restrict__ B,
                      int32_t *__restrict__ C, int M, int N, int K) {
    int8_t *sA[2] = { smem, smem + 2 * BM * APITCH };
    int8_t *sB[2] = { smem + BM * APITCH, smem + BM * APITCH + BK * BPITCH };
    // stage size = BM*APITCH + BK*BPITCH
    const int STAGE = BM * APITCH + BN * BPITCH;
    sA[1] = smem + STAGE; sB[1] = smem + STAGE + BM * APITCH;

    const int tid = threadIdx.y * 32 + threadIdx.x;   // blockDim (32,8) = 256 thr
    const int warp = tid / 32;                        // 0..7
    const int wr = warp % 2, wc = warp / 2;           // 2 rows x 4 cols of warps
    const int lane = threadIdx.x;

    const int bm = blockIdx.y * BM, bn = blockIdx.x * BN;
    const int warpM = bm + wr * 64;                   // 64 tall per warp
    const int warpN = bn + wc * 32;                   // 32 wide per warp

    // ---- async tile loader: 128x64 A + 64x128 B per stage
    // A: 128 rows x 64 cols = 8 x 16B per row -> 1024 chunks; 4 chunks/thread
    auto load_stage = [&](int buf, int k0) {
        // A tile
        for (int c = tid; c < BM * (BK / 16); c += 256) {
            int r = c / (BK / 16), col = (c % (BK / 16)) * 16;
            unsigned d = __cvta_generic_to_shared(&sA[buf][r * APITCH + col]);
            cp16(d, A + (long)(bm + r) * K + k0 + col);
        }
        // B tile (global B transposed: [N][K])
        for (int c = tid; c < BN * (BK / 16); c += 256) {
            int r = c / (BK / 16), col = (c % (BK / 16)) * 16;
            unsigned d = __cvta_generic_to_shared(&sB[buf][r * BPITCH + col]);
            cp16(d, B + (long)(bn + r) * K + k0 + col);
        }
        cp_commit();
    };

    int32_t acc[4][4][4];  // 4 m-tiles x 4 n-frag? warp tile 64x32: 4 m16 x 4 n8
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            for (int r = 0; r < 4; r++) acc[i][j][r] = 0;

    load_stage(0, 0);
#ifdef DBG
    cp_wait0(); __syncthreads();
    if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0)
        printf("dbg A0=%d B0(n0k0)=%d sA00=%d sB00=%d sB08=%d\n", (int)A[0], (int)B[0], (int)sA[0][0], (int)sB[0][0], (int)sB[0][8]);
    if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) printf("dbg after-load ok\n");
#endif

    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        int kn = k0 + BK;
        if (kn < K) load_stage(buf ^ 1, kn); else cp_commit();
        if (kn < K) cp_wait1(); else cp_wait0();
        __syncthreads();

        // per k16 in BK
        for (int kk = 0; kk < BK; kk += 32) {
            // load A frags: 4 m16x32 tiles -> ldmatrix.x4 twice each (16x16 halves)? A m16k32 = 512B = x4 once
            uint32_t af[4][4];
            for (int i = 0; i < 4; i++) {
                int row = wr * 64 + i * 16 + (lane % 16);
                int colb = kk + (lane / 16) * 16;   // two halves of 16 bytes
                unsigned s = __cvta_generic_to_shared(&sA[buf][row * APITCH + colb]);
                ldm_x4(s, af[i][0], af[i][1], af[i][2], af[i][3], false);
            }
            // B frags canonical for m16n8k32.col: x4 matrices = {n0-7 k0-15, n0-7 k16-31, n8-15 k0-15, n8-15 k16-31}
            uint32_t bf[2][4];
            {
                int grp = lane / 8, r8 = lane % 8;
                int koff = (grp % 2) * 16;
                for (int j = 0; j < 2; j++) {
                    int nrow = wc * 32 + j * 16 + (grp >= 2 ? 8 : 0) + r8;
                    unsigned s = __cvta_generic_to_shared(&sB[buf][nrow * BPITCH + kk + koff]);
                    ldm_x4(s, bf[j][0], bf[j][1], bf[j][2], bf[j][3], false);
                }
            }
            for (int i = 0; i < 4; i++) {
                mma_s8(acc[i][0][0], acc[i][0][1], acc[i][0][2], acc[i][0][3],
                       af[i][0], af[i][1], af[i][2], af[i][3], bf[0][0], bf[0][1]);
                mma_s8(acc[i][1][0], acc[i][1][1], acc[i][1][2], acc[i][1][3],
                       af[i][0], af[i][1], af[i][2], af[i][3], bf[0][2], bf[0][3]);
                mma_s8(acc[i][2][0], acc[i][2][1], acc[i][2][2], acc[i][2][3],
                       af[i][0], af[i][1], af[i][2], af[i][3], bf[1][0], bf[1][1]);
                mma_s8(acc[i][3][0], acc[i][3][1], acc[i][3][2], acc[i][3][3],
                       af[i][0], af[i][1], af[i][2], af[i][3], bf[1][2], bf[1][3]);
            }
        }
        __syncthreads();
        buf ^= 1;
    }

#ifdef DBG
    if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0)
        printf("dbg acc[0][0]=%d %d %d %d\n", acc[0][0][0], acc[0][0][1], acc[0][0][2], acc[0][0][3]);
#endif
    // epilogue: store acc (fragment m16n8 layout) via smem to keep it simple in v1
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            int row0 = warpM + i * 16 + lane / 4;
            int col = warpN + j * 8 + (lane % 4) * 2;
            C[(long)row0 * N + col] = acc[i][j][0];
            C[(long)row0 * N + col + 1] = acc[i][j][1];
            C[(long)(row0 + 8) * N + col] = acc[i][j][2];
            C[(long)(row0 + 8) * N + col + 1] = acc[i][j][3];
        }
}

static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec / 1e9; }

int main(int argc, char **argv) {
    int M = argc > 3 ? atoi(argv[3]) : 4096, N = M, K = M;
    int iters = argc > 1 ? atoi(argv[1]) : 30;
    size_t szA = (size_t)M * K, szB = (size_t)K * N, szC = (size_t)M * N * 4;
    int8_t *A = (int8_t *)malloc(szA), *B = (int8_t *)malloc(szB);
    int32_t *C = (int32_t *)malloc(szC), *R = (int32_t *)malloc(szC);
    srand(42);
    for (size_t i = 0; i < szA; i++) A[i] = (int8_t)(rand() % 127);
    for (int r = 0; r < K; r++)
        for (int c = 0; c < N; c++) B[(long)c * K + r] = (int8_t)(rand() % 127);  // B stored [N][K]
    int8_t *dA, *dB; int32_t *dC;
    cudaMalloc(&dA, szA); cudaMalloc(&dB, szB); cudaMalloc(&dC, szC);
    cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, szB, cudaMemcpyHostToDevice);
    int shm = 2 * (BM * APITCH + BN * BPITCH);
    cudaFuncSetAttribute(gemm8, cudaFuncAttributeMaxDynamicSharedMemorySize, shm);
    dim3 block(32, 8), grid(N / BN, M / BM);
    gemm8<<<grid, block, shm>>>(dA, dB, dC, M, N, K);
    cudaError_t e = cudaDeviceSynchronize();
    printf("launch: %s\n", cudaGetErrorString(e));
    cudaMemcpy(C, dC, szC, cudaMemcpyDeviceToHost);
    printf("C[0]=%d C[1]=%d C[80]=%d A[0]=? A0*B0: \n", C[0], C[1], C[80]);

    // CPU reference check on 256x256x64 corner
    int T = 256;
    for (int r = 0; r < T; r += 37)
        for (int c = 0; c < T; c += 41) {
            int32_t s = 0;
            for (int k = 0; k < K; k++) s += (int32_t)A[(long)r * K + k] * B[(long)c * K + k];
            if (C[(long)r * N + c] != s) { printf("MISMATCH at %d,%d: got %d want %d\n", r, c, C[(long)r * N + c], s); return 1; }
        }
    printf("correctness: OK\n");

    double t0 = now();
    for (int it = 0; it < iters; it++) gemm8<<<grid, block, shm>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();
    double t = now() - t0;
    printf("ghost_v1 mma+cp.async: %dx%dx%d %.2f TOPS (%.2f ms/iter)\n", M, N, K,
           2.0 * M * N * K * iters / t / 1e12, t / iters * 1000);
    return 0;
}
