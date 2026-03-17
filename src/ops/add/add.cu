#include "../../core/Tensor.h"
#include <cuda_runtime.h>
#include <iostream>

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Accumulate gradient: grad_in += grad_out
__global__ void add_grad_kernel(const float* grad_out, float* grad_in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_in[idx] += grad_out[idx];
    }
}

void launch_add_cuda(const float* a, const float* b, float* c, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(a, b, c, n);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (add): " << cudaGetErrorString(err) << std::endl;
    }
}

void launch_add_grad_cuda(const float* grad_out, float* grad_in, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_grad_kernel<<<blocks, threads>>>(grad_out, grad_in, n);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (add_grad): " << cudaGetErrorString(err) << std::endl;
    }
}
