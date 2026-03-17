#include "ConcatGradFn.h"
#include <cstring>
#include <stdexcept>
#include <vector>
#include <iostream>

void launch_concat_grad_cuda(
    const float* grad_out,
    const std::vector<float*>& grad_in_ptrs,
    const std::vector<int>& input_cols,
    int rows,
    int out_cols,
    int tail
);

void ConcatGradFn::cpu_implementation(const Tensor grad_output) {
    Tensor go = grad_output;
    if (go.num_dims() < 2) {
        go = go.view(Shape({1, go.numel()}));
    }

    const Shape& out_shape = go.shape();

    size_t rows = out_shape[0];
    size_t out_cols = out_shape[1];

    // product of dims after dim 1
    size_t tail = 1;
    for (size_t d = 2; d < out_shape.num_dims(); ++d) {
        tail *= out_shape[d];
    }

    const float* grad_out_ptr = go.data();

    size_t col_offset = 0;

    for (auto& parent : parents_) {
        if (!parent.requires_grad()) {
            col_offset += parent.shape()[1];
            continue;
        }

        Tensor grad_in = parent.grad();
        float* grad_in_ptr = grad_in.data();

        size_t cols = parent.shape()[1];

        for (size_t r = 0; r < rows; ++r) {
            std::memcpy(
                grad_in_ptr + (r * cols) * tail,
                grad_out_ptr + (r * out_cols + col_offset) * tail,
                cols * tail * sizeof(float)
            );
        }

        col_offset += cols;
    }
}

void ConcatGradFn::cuda_implementation(const Tensor grad_output) {
    const Shape& out_shape = grad_output.shape();
    
    size_t rows = out_shape[0];
    size_t out_cols = out_shape[1];
    
    // Product of dims after dim 1
    size_t tail = 1;
    for (size_t d = 2; d < out_shape.num_dims(); ++d) {
        tail *= out_shape[d];
    }
    
    // Prepare gradient input pointers and column sizes
    std::vector<float*> grad_in_ptrs;
    std::vector<int> input_cols;
    
    for (auto& parent : parents_) {
        input_cols.push_back(parent.shape()[1]);
        
        if (parent.requires_grad()) {
            Tensor grad_in = parent.grad();
            grad_in_ptrs.push_back(grad_in.data());
        } else {
            grad_in_ptrs.push_back(nullptr);
        }
    }
    
    launch_concat_grad_cuda(
        grad_output.data(),
        grad_in_ptrs,
        input_cols,
        rows,
        out_cols,
        tail
    );
}
