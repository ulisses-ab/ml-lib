#include "../../core/Tensor.h"
#include <cuda_runtime.h>
#include <iostream>

__device__ float simple_rand_hash(unsigned long long seed, int idx) {
    unsigned long long state = seed + idx;
    // Wang Hash / SplitMix64 variant for stateless randomness
    state = (state ^ (state >> 30)) * 0xbf58476d1ce4e5b9ULL;
    state = (state ^ (state >> 27)) * 0x94d049bb133111ebULL;
    state = state ^ (state >> 31);
    
    // Use upper 32 bits and normalize
    unsigned int res = (unsigned int)(state >> 32);
    // [0, 1]
    float f = (float)res / 4294967295.0f;
    // [-0.1, 0.1]
    return f * 0.2f - 0.1f;
}

__global__ void rand_fill_kernel(float* data, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = simple_rand_hash(seed, idx);
    }
}

void launch_rand_fill_cuda(float* data, int n, unsigned long long seed) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    rand_fill_kernel<<<blocks, threads>>>(data, n, seed);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (rand_fill): " << cudaGetErrorString(err) << std::endl;
    }
}
