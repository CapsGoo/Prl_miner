// frag_test: validate ldmatrix + mma.sync m16n8k32 s8 fragment mapping (1 block, 64x64x64)
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

// global A: [M][K] row-major; global B stored transposed [N][K]
// smem A: row pitch 64+16=80 ; smem B: same
__global__ void k64(const int8_t *A, const int8_t *B, int32_t *C) {
    __shared__ int8_t sA[64 * 80];
    __shared__ int8_t sB[64 * 80];
    const int tid = threadIdx.y * 32 + threadIdx.x; // (32,2)=64 threads? use (32,2) 2 warps
    // load A 64 rows x 64 cols; B(n) 64 rows x 64 cols(k)
    for (int c = tid; c < 64 * 4; c += 64) {
        int r = c / 4, col = (c % 4) * 16;
        sA[r * 80 + col] = A[r * 64 + col];
        sB[r * 80 + col] = B[r * 64 + col];
    }
    __syncthreads();

    const int warp = tid / 32, lane = tid % 32;   // 2 warps: warp0 covers m0-31, warp1 m32-63
    const int wm = warp * 32;

    int32_t acc[2][4][4];
    for (int i = 0; i < 2; i++) for (int j = 0; j < 4; j++) for (int r = 0; r < 4; r++) acc[i][j][r] = 0;

    for (int kk = 0; kk < 64; kk += 32) {
        uint32_t af[2][4];
        for (int i = 0; i < 2; i++) {
            int row = wm + i * 16 + (lane % 16);
            int colb = kk + (lane / 16) * 16;
            unsigned s = __cvta_generic_to_shared(&sA[row * 80 + colb]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(af[i][0]), "=r"(af[i][1]), "=r"(af[i][2]), "=r"(af[i][3]) : "r"(s));
        }
        uint32_t bf[2][4];   // two n16 tiles (n0-15, n16-31... we need 64 cols? per warp: all 64 n? simplify: warp covers 64x32)
        // warp computes n 0..63 -> four n8 tiles = two n16 loads
        for (int j = 0; j < 2; j++) {
            int row = j * 32 + (lane % 16);          // n rows of B[N][K]: n0-31 chunk0 (registers hold n16 pairs?)
            int colb = kk + (lane / 16) * 16;
            unsigned s = __cvta_generic_to_shared(&sB[row * 80 + colb]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(bf[j][0]), "=r"(bf[j][1]), "=r"(bf[j][2]), "=r"(bf[j][3]) : "r"(s));
        }
        for (int i = 0; i < 2; i++) {
            // n8 tiles: from bf[0]: (0,1)->n0-7? (2,3)->n8-15? test both mappings by using j loop of 4 tiles:
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+r"(acc[i][0][0]), "+r"(acc[i][0][1]), "+r"(acc[i][0][2]), "+r"(acc[i][0][3])
                : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]), "r"(bf[0][0]), "r"(bf[0][2]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+r"(acc[i][1][0]), "+r"(acc[i][1][1]), "+r"(acc[i][1][2]), "+r"(acc[i][1][3])
                : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]), "r"(bf[0][1]), "r"(bf[0][3]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+r"(acc[i][2][0]), "+r"(acc[i][2][1]), "+r"(acc[i][2][2]), "+r"(acc[i][2][3])
                : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]), "r"(bf[1][0]), "r"(bf[1][2]));
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+r"(acc[i][3][0]), "+r"(acc[i][3][1]), "+r"(acc[i][3][2]), "+r"(acc[i][3][3])
                : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]), "r"(bf[1][1]), "r"(bf[1][3]));
        }
    }

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 4; j++) {
            int row = wm + i * 16 + lane / 4;
            int col = j * 8 + (lane % 4) * 2;
            C[row * 64 + col] = acc[i][j][0];
            C[row * 64 + col + 1] = acc[i][j][1];
            C[(row + 8) * 64 + col] = acc[i][j][2];
            C[(row + 8) * 64 + col + 1] = acc[i][j][3];
        }
}

int main() {
    int8_t *A = (int8_t *)malloc(64 * 64), *B = (int8_t *)malloc(64 * 64);
    int32_t *C = (int32_t *)malloc(64 * 64 * 4);
    srand(1);
    for (int i = 0; i < 4096; i++) { A[i] = (int8_t)(rand() % 101 - 50); B[i] = (int8_t)(rand() % 101 - 50); }
    int8_t *dA, *dB; int32_t *dC;
    cudaMalloc(&dA, 4096); cudaMalloc(&dB, 4096); cudaMalloc(&dC, 4096 * 4);
    cudaMemcpy(dA, A, 4096, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, 4096, cudaMemcpyHostToDevice);
    k64<<<1, dim3(32, 2)>>>(dA, dB, dC);
    cudaError_t e = cudaDeviceSynchronize();
    if (e) { printf("ERR: %s\n", cudaGetErrorString(e)); return 1; }
    cudaMemcpy(C, dC, 4096 * 4, cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int r = 0; r < 64; r++)
        for (int c = 0; c < 64; c++) {
            int32_t s = 0;
            for (int k = 0; k < 64; k++) s += (int32_t)A[r * 64 + k] * B[c * 64 + k];  // B[N][K]
            if (C[r * 64 + c] != s) {
                if (bad < 6) printf("bad(%d,%d) got %d want %d\n", r, c, C[r * 64 + c], s);
                bad++;
            }
        }
    printf(bad ? "FAIL %d\n" : "FRAGMENTS OK\n", bad);
    return 0;
}
