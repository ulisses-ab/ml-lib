#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cstdio> // for printing if needed

// Helper for block reduction
template<typename T>
__inline__ __device__ T warpReduceMax(T val) {
    for (int offset = 32 / 2; offset > 0; offset /= 2)
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

template<typename T>
__inline__ __device__ T blockReduceMax(T val) {
    static __shared__ T shared[32]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceMax(val);     // Each warp performs partial reduction

    if (lane == 0) shared[wid] = val; // Write reduced value to shared mem

    __syncthreads();              // Wait for all partial reductions

    // Read from shared mem and perform final reduction
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -1e38f; // conservative min

    if (wid == 0) val = warpReduceMax(val); // Final reduction within first warp

    return val;
}

template<typename T>
__inline__ __device__ T warpReduceSum(T val) {
    for (int offset = 32 / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

template<typename T>
__inline__ __device__ T blockReduceSum(T val) {
    static __shared__ T shared[32]; 
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceSum(val);

    if (lane == 0) shared[wid] = val;

    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;

    if (wid == 0) val = warpReduceSum(val);

    return val;
}

__global__ void softmax_kernel(const float* A, float* B, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    // 1. Find Max
    float max_val = -1e38f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        max_val = max(max_val, A[row * cols + i]);
    }
    max_val = blockReduceMax(max_val);
    __shared__ float shared_max;
    if (threadIdx.x == 0) shared_max = max_val;
    __syncthreads();
    max_val = shared_max;

    // 2. Compute Sum of Exps
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum += expf(A[row * cols + i] - max_val);
    }
    sum = blockReduceSum(sum);
    __shared__ float shared_sum;
    if (threadIdx.x == 0) shared_sum = sum;
    __syncthreads();
    sum = shared_sum;

    // 3. Compute Softmax and Write
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        B[row * cols + i] = expf(A[row * cols + i] - max_val) / sum;
    }
}

void launch_softmax_cuda(const float* A, float* B, int rows, int cols) {
    int threads_per_block = 256;
    while(threads_per_block < cols && threads_per_block < 1024) {
         threads_per_block *= 2;
    }
    
    // One block per row
    dim3 grid(rows);
    dim3 block(threads_per_block);

    // Shared memory logic handles by static allocation inside helper functions for now, 
    // or we can use dynamic if needed. The helpers use static shared[32], which is fine for block size <= 1024.
    
    softmax_kernel<<<grid, block>>>(A, B, rows, cols);
    cudaDeviceSynchronize();
}

__global__ void softmax_grad_kernel(const float* grad_output, const float* input, float* grad_input, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    // We definitely need to recompute Softmax(input) first because we don't have it stored.
    // Or we could have passed it if we cached it. But SoftmaxGradFn signature in CPU impl implies recompute.
    // Steps:
    // 1. Find Max of input
    float max_val = -1e38f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        max_val = max(max_val, input[row * cols + i]);
    }
    max_val = blockReduceMax(max_val);
    __shared__ float shared_max;
    if (threadIdx.x == 0) shared_max = max_val;
    __syncthreads();
    max_val = shared_max;

    // 2. Compute Sum of Exps
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum += expf(input[row * cols + i] - max_val);
    }
    sum = blockReduceSum(sum);
    __shared__ float shared_sum;
    if (threadIdx.x == 0) shared_sum = sum;
    __syncthreads();
    sum = shared_sum;
    
    // 3. Compute dot = sum(grad_output * softmax)
    float dot = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float y = expf(input[row * cols + i] - max_val) / sum;
        dot += grad_output[row * cols + i] * y;
    }
    dot = blockReduceSum(dot);
    __shared__ float shared_dot;
    if (threadIdx.x == 0) shared_dot = dot;
    __syncthreads();
    dot = shared_dot;

    // 4. Compute gradient
    // dx = y * (g - dot)
    // We update grad_input (accumulate)
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float y = expf(input[row * cols + i] - max_val) / sum;
        grad_input[row * cols + i] += y * (grad_output[row * cols + i] - dot);
    }
}

void launch_softmax_grad_cuda(const float* grad_output, const float* input, float* grad_input, int rows, int cols) {
    int threads_per_block = 256;
    while(threads_per_block < cols && threads_per_block < 1024) {
         threads_per_block *= 2;
    }
    
    dim3 grid(rows);
    dim3 block(threads_per_block);

    softmax_grad_kernel<<<grid, block>>>(grad_output, input, grad_input, rows, cols);
    cudaDeviceSynchronize();
}
