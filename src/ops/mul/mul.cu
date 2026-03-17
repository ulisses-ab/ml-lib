#include <cuda_runtime.h>
#include <iostream>

__global__ void mul_kernel(const float* A, float* B, float val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        B[idx] = A[idx] * val;
    }
}

__global__ void mul_grad_kernel(float* grad_input, const float* grad_output, float val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_input[idx] += grad_output[idx] * val;
    }
}

__global__ void mul_tensor_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

__global__ void mul_grad_accumulate_kernel(float* grad_in, const float* grad_out, const float* other, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_in[idx] += grad_out[idx] * other[idx];
    }
}

void launch_mul_cuda(const float* A, float* B, float val, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    mul_kernel<<<blocks, threads>>>(A, B, val, n);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (mul): " << cudaGetErrorString(err) << std::endl;
    }
}

void launch_mul_grad_cuda(float* grad_input, const float* grad_output, float val, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    mul_grad_kernel<<<blocks, threads>>>(grad_input, grad_output, val, n);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (mul_grad): " << cudaGetErrorString(err) << std::endl;
    }
}

void launch_mul_tensor_cuda(const float* a, const float* b, float* c, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    mul_tensor_kernel<<<blocks, threads>>>(a, b, c, n);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (mul_tensor): " << cudaGetErrorString(err) << std::endl;
    }
}

void launch_mul_grad_accumulate_cuda(float* grad_in, const float* grad_out, const float* other, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    mul_grad_accumulate_kernel<<<blocks, threads>>>(grad_in, grad_out, other, n);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (mul_grad_accumulate): " << cudaGetErrorString(err) << std::endl;
    }
}
