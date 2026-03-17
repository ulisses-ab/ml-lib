#include <cuda_runtime.h>
#include <iostream>


__global__ void div_tensor_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] / b[idx];
    }
}


void launch_div_tensor_cuda(const float* a, const float* b, float* c, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    div_tensor_kernel<<<blocks, threads>>>(a, b, c, n);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (mul): " << cudaGetErrorString(err) << std::endl;
    }
}

__global__ void div_grad_accumulate_kernel(
    float* a_grad,
    float* b_grad,
    const float* a,
    const float* b,
    const float* grad,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float ai = a[idx];
    float bi = b[idx];
    float go = grad[idx];

    // ∂L/∂a += go * (1 / b)
    if (a_grad) {
        a_grad[idx] += go / bi;
    }

    // ∂L/∂b += go * (-a / b^2)
    if (b_grad) {
        b_grad[idx] += -go * ai / (bi * bi);
    }
}

void launch_div_grad_accumulate_cuda(
    float* a_grad,
    float* b_grad,
    const float* a,
    const float* b,
    const float* grad,
    int n
) {
    constexpr int BLOCK_SIZE = 256;
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    div_grad_accumulate_kernel<<<grid, BLOCK_SIZE>>>(
        a_grad,
        b_grad,
        a,
        b,
        grad,
        n
    );
}
