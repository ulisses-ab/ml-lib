#include "../../core/Tensor.h"
#include <cuda_runtime.h>
#include <iostream>

__global__ void relu_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        out[idx] = val > 0.0f ? val : 0.0f;
    }
}

__global__ void relu_backward_kernel(const float* in, const float* grad_out, float* grad_in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (in[idx] > 0.0f) {
            // Using atomicAdd for safety if multiple streams/threads touched this (though unlikely for elementwise)
            // But standard Autograd practice is accumulation.
            // Since this is element-wise 1-to-1, standard addition is fine if we assume no race condition on the exact same index from the exact same kernel launch.
            // However, we are writing to 'grad_in'.
            // For safety and correctness with existing gradients:
            grad_in[idx] += grad_out[idx];
        }
        // If <= 0, add 0, which means do nothing.
    }
}

void launch_relu_cuda(const float* in, float* out, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(in, out, n);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (relu): " << cudaGetErrorString(err) << std::endl;
    }
}

void launch_relu_backward_cuda(const float* in, const float* grad_out, float* grad_in, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu_backward_kernel<<<blocks, threads>>>(in, grad_out, grad_in, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (relu_backward): " << cudaGetErrorString(err) << std::endl;
    }
}
