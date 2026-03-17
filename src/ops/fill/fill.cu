#include "../../core/Tensor.h"
#include <cuda_runtime.h>
#include <iostream>

__global__ void fill_kernel(float* data, float val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = val;
    }
}

void launch_fill_cuda(float* data, float val, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    fill_kernel<<<blocks, threads>>>(data, val, n);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (fill): " << cudaGetErrorString(err) << std::endl;
    }
}
