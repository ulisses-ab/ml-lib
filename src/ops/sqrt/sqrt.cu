#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

__global__ void sqrt_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = sqrtf(x[idx]);
    }
}

__global__ void sqrt_grad_kernel(float* grad_x, const float* grad_y, const float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // dL/dx = dL/dy * (1 / (2 * sqrt(x)))
        float val = x[idx];
        if (val > 1e-12f) { // Avoid division by zero
            grad_x[idx] += grad_y[idx] * (0.5f / sqrtf(val));
        } else {
             // Technically infinite gradient at 0, usually 0 or ignored in practice for stability or clamped.
             // For now let's just do 0 if too small to avoid NaNs if val is 0.
        }
    }
}

void launch_sqrt_cuda(const float* x, float* y, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    sqrt_kernel<<<blocks, threads>>>(x, y, n);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (sqrt): " << cudaGetErrorString(err) << std::endl;
    }
}

void launch_sqrt_grad_cuda(float* grad_x, const float* grad_y, const float* x, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    sqrt_grad_kernel<<<blocks, threads>>>(grad_x, grad_y, x, n);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (sqrt_grad): " << cudaGetErrorString(err) << std::endl;
    }
}
