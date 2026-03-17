#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>

__global__ void layer_norm_kernel(
    const float* x,
    const float* gamma,
    const float* beta,
    float* output,
    int N,
    int D,
    float eps
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) {
        // Compute mean for row i
        float mean = 0.0f;
        for (int j = 0; j < D; ++j) {
            mean += x[i * D + j];
        }
        mean /= static_cast<float>(D);

        // Compute variance for row i
        float variance = 0.0f;
        for (int j = 0; j < D; ++j) {
            float diff = x[i * D + j] - mean;
            variance += diff * diff;
        }
        variance /= static_cast<float>(D);

        // Normalize and apply affine transformation
        float inv_std = 1.0f / sqrtf(variance + eps);
        for (int j = 0; j < D; ++j) {
            float normalized = (x[i * D + j] - mean) * inv_std;
            output[i * D + j] = gamma[j] * normalized + beta[j];
        }
    }
}

void launch_layer_norm_cuda(
    const float* x,
    const float* gamma,
    const float* beta,
    float* output,
    int N,
    int D,
    float eps
) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    layer_norm_kernel<<<blocks, threads>>>(x, gamma, beta, output, N, D, eps);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error (layer_norm): %s\n", cudaGetErrorString(err));
    }
}

__global__ void layer_norm_grad_kernel(
    const float* grad_out,
    const float* x,
    const float* gamma,
    float* grad_x,
    float* grad_gamma,
    float* grad_beta,
    int N,
    int D,
    float eps
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) {
        // Recompute mean and variance
        float mean = 0.0f;
        for (int j = 0; j < D; ++j) {
            mean += x[i * D + j];
        }
        mean /= static_cast<float>(D);

        float variance = 0.0f;
        for (int j = 0; j < D; ++j) {
            float diff = x[i * D + j] - mean;
            variance += diff * diff;
        }
        variance /= static_cast<float>(D);
        float inv_std = 1.0f / sqrtf(variance + eps);
        float inv_D = 1.0f / static_cast<float>(D);

        // Compute normalized values and accumulate gradients
        float sum_grad_out = 0.0f;
        float sum_grad_norm = 0.0f;
        
        for (int j = 0; j < D; ++j) {
            float normalized = (x[i * D + j] - mean) * inv_std;
            float g_out = grad_out[i * D + j];
            
            sum_grad_out += g_out;
            sum_grad_norm += g_out * normalized;
        }

        // Compute gradients
        for (int j = 0; j < D; ++j) {
            float normalized = (x[i * D + j] - mean) * inv_std;
            float g_out = grad_out[i * D + j];
            
            // grad_gamma (if needed)
            if (grad_gamma != nullptr) {
                atomicAdd(&grad_gamma[j], g_out * normalized);
            }
            
            // grad_beta (if needed)
            if (grad_beta != nullptr) {
                atomicAdd(&grad_beta[j], g_out);
            }
            
            // grad_x (if needed)
            if (grad_x != nullptr) {
                float term1 = g_out * inv_D;
                float term2 = normalized * sum_grad_norm * inv_D;
                atomicAdd(&grad_x[i * D + j], gamma[j] * inv_std * (g_out - term1 - term2));
            }
        }
    }
}

void launch_layer_norm_grad_cuda(
    const float* grad_out,
    const float* x,
    const float* gamma,
    float* grad_x,
    float* grad_gamma,
    float* grad_beta,
    int N,
    int D,
    float eps
) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    layer_norm_grad_kernel<<<blocks, threads>>>(
        grad_out, x, gamma, grad_x, grad_gamma, grad_beta, N, D, eps
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error (layer_norm_grad): %s\n", cudaGetErrorString(err));
    }
}


