#include <cuda_runtime.h>
#include <stdexcept>

__global__ void repeat_kernel(
    const float* src,
    float* dst,
    int rows,
    int repeats
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < rows && c < repeats) {
        dst[c * rows + r] = src[r];
    }
}

void launch_repeat_cuda(const float* src, float* dst, int src_size, int repeats) {
    if (!src || !dst) {
        throw std::invalid_argument("repeat cuda: null pointer");
    }

    dim3 block(256, 1);
    dim3 grid(
        (src_size + block.x - 1) / block.x,
        repeats
    );

    repeat_kernel<<<grid, block>>>(src, dst, src_size, repeats);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

__global__ void repeat_grad_kernel(
    const float* grad_out,
    float* grad_in,
    int rows,
    int repeats
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < rows) {
        float sum = 0.0f;
        for (int c = 0; c < repeats; ++c) {
            sum += grad_out[c * rows + r];
        }
        grad_in[r] += sum;
    }
}

void launch_repeat_grad_cuda(
    const float* grad_out,
    float* grad_in,
    int rows,
    int repeats
) {
    if (!grad_out || !grad_in) {
        throw std::invalid_argument("repeat grad cuda: null pointer");
    }

    const int threads = 256;
    const int blocks = (rows + threads - 1) / threads;

    repeat_grad_kernel<<<blocks, threads>>>(
        grad_out,
        grad_in,
        rows,
        repeats
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}
