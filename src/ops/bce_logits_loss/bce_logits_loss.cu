#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <algorithm>

__global__ void bce_logits_loss_kernel(const float* input, const float* target, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float z = target[idx];
        
        // loss = max(x, 0) - x * z + log(1 + exp(-abs(x)))
        float max_val = (x > 0.0f) ? x : 0.0f;
        float neg_abs_x = (x > 0.0f) ? -x : x; 
        
        float val = max_val - x * z + logf(1.0f + expf(neg_abs_x));
        
        atomicAdd(output, val / static_cast<float>(n));
    }
}

void launch_bce_logits_loss_cuda(const float* input, const float* target, float* output, int n) {
    cudaMemset(output, 0, sizeof(float));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    bce_logits_loss_kernel<<<blocks, threads>>>(input, target, output, n);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error (bce_logits_loss): %s\n", cudaGetErrorString(err));
    }
}

__global__ void bce_logits_loss_grad_kernel(const float* input, const float* target, const float* grad_output, float* grad_input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float z = target[idx];
        
        // dL/dx = sigmoid(x) - z
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        
        float g = sigmoid_x - z;
        float upstream = grad_output[0];
        
        atomicAdd(&grad_input[idx], g * upstream / static_cast<float>(n));
    }
}

void launch_bce_logits_loss_grad_cuda(const float* input, const float* target, const float* grad_output, float* grad_input, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    bce_logits_loss_grad_kernel<<<blocks, threads>>>(input, target, grad_output, grad_input, n);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error (bce_logits_loss_grad): %s\n", cudaGetErrorString(err));
    }
}
