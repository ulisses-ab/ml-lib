#include <stdio.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <math.h>
#include "../../core/Tensor.h"

#define CEILDIV(a, b) ((a + b - 1) / b)
#define INDEX_2D(row, col, width) ((row) * (width) + col)

constexpr int MULTIPLICATION_TILE_SIZE = 32;

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int B_col = threadIdx.x + blockIdx.x * blockDim.x;
    int A_row = threadIdx.y + blockIdx.y * blockDim.y;

    __shared__ float tile_A[MULTIPLICATION_TILE_SIZE][MULTIPLICATION_TILE_SIZE];
    __shared__ float tile_B[MULTIPLICATION_TILE_SIZE][MULTIPLICATION_TILE_SIZE];

    float sum = 0;

    for(int tileIdx = 0; tileIdx < (N + MULTIPLICATION_TILE_SIZE - 1) / MULTIPLICATION_TILE_SIZE; tileIdx++) {
        int col = tileIdx * MULTIPLICATION_TILE_SIZE + threadIdx.x;
        int row = tileIdx * MULTIPLICATION_TILE_SIZE + threadIdx.y;

        if (A_row < M && col < N)
            tile_A[threadIdx.y][threadIdx.x] = A[A_row * N + col];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (row < N && B_col < K)
            tile_B[threadIdx.y][threadIdx.x] = B[row * K + B_col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for(int i = 0; i < MULTIPLICATION_TILE_SIZE; i++) {
            sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (A_row < M && B_col < K)
        C[A_row * K + B_col] = sum;
}

void multiply(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(MULTIPLICATION_TILE_SIZE, MULTIPLICATION_TILE_SIZE);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

void launch_matmul_cuda(const Tensor& a, const Tensor& b, Tensor& c) {
    int M = a.shape()[0];
    int N = a.shape()[1];
    int K = b.shape()[1];

    multiply(a.data(), b.data(), c.data(), M, N, K);
}
