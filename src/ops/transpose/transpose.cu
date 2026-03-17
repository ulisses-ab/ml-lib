#include <cuda_runtime.h>
#include <iostream>

__global__ void transpose_2d_kernel(const float* src, float* dst, int rows, int cols) {
    // rows = src rows
    // cols = src cols
    // src shape (rows, cols)
    // dst shape (cols, rows)
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        // x corresponds to col index in src (0..cols-1) -> row index in dst
        // y corresponds to row index in src (0..rows-1) -> col index in dst
        
        int src_idx = y * cols + x;
        int dst_idx = x * rows + y;
        
        dst[dst_idx] = src[src_idx];
    }
}

void launch_transpose_2d_cuda(const float* src, float* dst, int rows, int cols) {
    dim3 dimBlock(32, 32);
    dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);
    transpose_2d_kernel<<<dimGrid, dimBlock>>>(src, dst, rows, cols);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (transpose): " << cudaGetErrorString(err) << std::endl;
    }
}

__global__ void add_inplace_kernel(float* dst, const float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] += src[idx];
    }
}

void launch_add_accumulate_cuda(float* dst, const float* src, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_inplace_kernel<<<blocks, threads>>>(dst, src, n);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (transpose_add): " << cudaGetErrorString(err) << std::endl;
    }
}
