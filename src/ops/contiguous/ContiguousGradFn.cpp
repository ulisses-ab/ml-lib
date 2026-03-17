#include "ContiguousGradFn.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <stdexcept>

void ContiguousGradFn::cpu_implementation(const Tensor grad_output) {
    if (parents_.size() != 1) {
        throw std::logic_error("contiguous grad: expected exactly one parent");
    }

    Tensor input = parents_[0];

    if (!input.requires_grad()) {
        return;
    }

    Tensor grad_input = input.grad();

    const Shape& shape = grad_output.shape();
    int total_elements = shape.numel();
    int rank = shape.num_dims();

    std::vector<int> indices(rank);

    const float* grad_out_data = grad_output.data();

    for (int i = 0; i < total_elements; ++i) {
        int tmp = i;

        // Convert flat index -> multi-dimensional index
        for (int d = rank - 1; d >= 0; --d) {
            indices[d] = tmp % shape[d];
            tmp /= shape[d];
        }

        // Accumulate into grad_input using original strides
        grad_input.at(indices) += grad_out_data[i];
    }
}

void launch_contiguous_2d_backward_cuda(
    const float* grad_output,
    float* grad_input,
    int M,
    int N,
    int row_stride,
    int col_stride
);

void ContiguousGradFn::cuda_implementation(const Tensor grad_output) {
    if (grad_output.num_dims() == 2) {
        launch_contiguous_2d_backward_cuda(
            grad_output.data(), 
            parents_[0].raw_data(), 
            grad_output.shape()[0],
            grad_output.shape()[1],
            grad_output.layout().stride(0),
            grad_output.layout().stride(1)
        );
        return;
    } 

    throw std::logic_error("cuda contiguous grad fn not implemented for this number of dimensions");
}