#include "TransposeGradFn.h"
#include "transpose.h"
#include <stdexcept>
#include <vector>

void launch_add_accumulate_cuda(float* dst, const float* src, int n);

void TransposeGradFn::cpu_implementation(const Tensor grad_output) {
    if (parents_.size() != 1) return;
    Tensor input = parents_[0];
    if (!input.requires_grad()) return;
    
    Tensor grad_input = input.grad();

    // grad_output is dL/d(Output). Output = A^T.
    // We want dL/dA = (dL/d(Output))^T.
    // So we transpose grad_output with same dims.
    Tensor restored_grad = transpose(grad_output, dim0_, dim1_);

    // Accumulate into grad_input
    // Assuming both are contiguous sequences of same length
    float* dst = grad_input.data();
    const float* src = restored_grad.data();
    int n = grad_input.numel();

    for(int i=0; i<n; ++i) {
        dst[i] += src[i];
    }
}

void TransposeGradFn::cuda_implementation(const Tensor grad_output) {
    if (parents_.size() != 1) return;
    Tensor input = parents_[0];
    if (!input.requires_grad()) return;
    
    Tensor grad_input = input.grad();
    
    // This will run the forward transpose kernel on the gradient
    Tensor restored_grad = transpose(grad_output, dim0_, dim1_);

    if (!grad_input.is_contiguous()) {
        throw std::logic_error("TransposeGradFn: grad_input must be contiguous for now");
    }

    launch_add_accumulate_cuda(grad_input.data(), restored_grad.data(), grad_input.numel());
}
