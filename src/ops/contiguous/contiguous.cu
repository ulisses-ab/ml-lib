#include <cuda_runtime.h>
#include <stdexcept>

__global__ void contiguous_2d_kernel(
    const float* src,
    float* dst,
    int M,
    int N,
    int row_stride,
    int col_stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;

    if (idx >= total) return;

    int i = idx / N;
    int j = idx % N;

    dst[idx] = src[i * row_stride + j * col_stride];
}

void launch_contiguous_2d_cuda(
    const float* src,
    float* dst,
    int M,
    int N,
    int row_stride,
    int col_stride
) {
    if (!src || !dst) {
        throw std::invalid_argument("launch_contiguous_2d_cuda: null pointer");
    }

    int total = M * N;
    if (total == 0) return;

    constexpr int BLOCK_SIZE = 256;
    int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    contiguous_2d_kernel<<<grid, BLOCK_SIZE>>>(
        src,
        dst,
        M,
        N,
        row_stride,
        col_stride
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

__global__ void contiguous_2d_backward_kernel(
    const float* grad_output,
    float* grad_input,
    int M,
    int N,
    int row_stride,
    int col_stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;

    if (idx >= total) return;

    int i = idx / N;
    int j = idx % N;

    atomicAdd(
        &grad_input[i * row_stride + j * col_stride],
        grad_output[idx]
    );
}

void launch_contiguous_2d_backward_cuda(
    const float* grad_output,
    float* grad_input,
    int M,
    int N,
    int row_stride,
    int col_stride
) {
    if (!grad_output || !grad_input) {
        throw std::invalid_argument(
            "launch_contiguous_2d_backward_cuda: null pointer"
        );
    }

    int total = M * N;
    if (total == 0) return;

    constexpr int BLOCK_SIZE = 256;
    int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    contiguous_2d_backward_kernel<<<grid, BLOCK_SIZE>>>(
        grad_output,
        grad_input,
        M,
        N,
        row_stride,
        col_stride
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}
