#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>

__global__ void concat_kernel(
    const float* input,
    float* output,
    int input_cols,
    int col_offset,
    int rows,
    int out_cols,
    int tail
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = rows * input_cols * tail;
    
    if (idx < total_elements) {
        // Calculate position in input tensor
        int tail_idx = idx % tail;
        int col_row_idx = idx / tail;
        int col = col_row_idx % input_cols;
        int row = col_row_idx / input_cols;
        
        // Calculate position in output tensor
        int out_col = col_offset + col;
        int out_idx = (row * out_cols + out_col) * tail + tail_idx;
        
        // Copy element
        output[out_idx] = input[idx];
    }
}

void launch_concat_cuda(
    const std::vector<const float*>& input_ptrs,
    float* output,
    const std::vector<int>& input_cols,
    int rows,
    int out_cols,
    int tail
) {
    int col_offset = 0;
    
    for (size_t i = 0; i < input_ptrs.size(); i++) {
        const float* input = input_ptrs[i];
        int cols = input_cols[i];
        int total_elements = rows * cols * tail;
        
        if (total_elements > 0) {
            int threads = 256;
            int blocks = (total_elements + threads - 1) / threads;
            
            concat_kernel<<<blocks, threads>>>(
                input, output, cols, col_offset, rows, out_cols, tail
            );
        }
        
        col_offset += cols;
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error (concat): %s\n", cudaGetErrorString(err));
    }
}

__global__ void concat_grad_kernel(
    const float* grad_out,
    float* grad_in,
    int input_cols,
    int col_offset,
    int rows,
    int out_cols,
    int tail
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = rows * input_cols * tail;
    
    if (idx < total_elements) {
        // Calculate position in input gradient tensor
        int tail_idx = idx % tail;
        int col_row_idx = idx / tail;
        int col = col_row_idx % input_cols;
        int row = col_row_idx / input_cols;
        
        // Calculate position in output gradient tensor
        int out_col = col_offset + col;
        int out_idx = (row * out_cols + out_col) * tail + tail_idx;
        
        // Copy element
        grad_in[idx] = grad_out[out_idx];
    }
}

void launch_concat_grad_cuda(
    const float* grad_out,
    const std::vector<float*>& grad_in_ptrs,
    const std::vector<int>& input_cols,
    int rows,
    int out_cols,
    int tail
) {
    int col_offset = 0;
    
    for (size_t i = 0; i < grad_in_ptrs.size(); i++) {
        float* grad_in = grad_in_ptrs[i];
        if (grad_in == nullptr) {
            col_offset += input_cols[i];
            continue;
        }
        
        int cols = input_cols[i];
        int total_elements = rows * cols * tail;
        
        if (total_elements > 0) {
            int threads = 256;
            int blocks = (total_elements + threads - 1) / threads;
            
            concat_grad_kernel<<<blocks, threads>>>(
                grad_out, grad_in, cols, col_offset, rows, out_cols, tail
            );
        }
        
        col_offset += cols;
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error (concat_grad): %s\n", cudaGetErrorString(err));
    }
}
