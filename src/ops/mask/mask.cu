#include <cuda_runtime.h>

__global__ void mask_kernel(const float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        for (int j = 0; j < n; j++) {
            if (i >= j) {
                b[i * n + j] = a[i * n + j];
            }
        }
    }
}

void launch_mask_cuda(const float* a, float* b, int n) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    mask_kernel<<<grid_size, block_size>>>(a, b, n);
}

__global__ void mask_grad_kernel(float* grad_a, const float* grad_out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        for (int j = 0; j < n; j++) {
            if (i >= j) {
                grad_a[i * n + j] += grad_out[i * n + j];
            }
        }
    }
}

void launch_mask_grad_cuda(float* grad_a, const float* grad_out, int n) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    mask_grad_kernel<<<grid_size, block_size>>>(grad_a, grad_out, n);
}