#include <cuda_runtime.h>
#include <iostream>

__global__ void gather_kernel(
    const int* indices, 
    const float* embedding, 
    float* output, 
    int N, 
    int D, 
    int V
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N * D) {
        int i = idx / D;  // which index in the sequence
        int j = idx % D;  // which dimension in the vector
        
        int embedding_idx = indices[i];
        
        if (embedding_idx >= 0 && embedding_idx < V) {
            output[idx] = embedding[embedding_idx * D + j];
        } else {
            // Out of bounds - set to 0 (or could throw error)
            output[idx] = 0.0f;
        }
    }
}

void launch_gather_cuda(
    const int* indices, 
    const float* embedding, 
    float* output, 
    int N, 
    int D, 
    int V
) {
    int threads = 256;
    int total_elements = N * D;
    int blocks = (total_elements + threads - 1) / threads;
    
    gather_kernel<<<blocks, threads>>>(indices, embedding, output, N, D, V);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error (gather): %s\n", cudaGetErrorString(err));
    }
}

__global__ void gather_grad_kernel(
    const int* indices,
    const float* grad_output,
    float* grad_embedding,
    int N,
    int D,
    int V
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N * D) {
        int i = idx / D;  // which index in the sequence
        int j = idx % D;  // which dimension in the vector
        
        int embedding_idx = indices[i];
        
        if (embedding_idx >= 0 && embedding_idx < V) {
            // Atomic add to handle potential collisions (same index used multiple times)
            atomicAdd(&grad_embedding[embedding_idx * D + j], grad_output[idx]);
        }
    }
}

void launch_gather_grad_cuda(
    const int* indices,
    const float* grad_output,
    float* grad_embedding,
    int N,
    int D,
    int V
) {
    int threads = 256;
    int total_elements = N * D;
    int blocks = (total_elements + threads - 1) / threads;
    
    gather_grad_kernel<<<blocks, threads>>>(indices, grad_output, grad_embedding, N, D, V);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (gather_grad): " << cudaGetErrorString(err) << std::endl;
    }
}

