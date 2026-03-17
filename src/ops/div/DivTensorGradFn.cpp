#include "DivTensorGradFn.h"
#include <stdexcept>
#include <vector>

void launch_div_grad_accumulate_cuda(
    float* a_grad, 
    float* b_grad, 
    const float* a, 
    const float* b, 
    const float* grad, 
    int n
);

void DivTensorGradFn::cpu_implementation(const Tensor grad_output) {
    if (parents_.size() != 2) {
        throw std::logic_error("MulTensorGradFn: expected 2 parents");
    }

    Tensor a = parents_[0];
    Tensor b = parents_[1];
    const float* go = grad_output.data();
    int n = grad_output.numel();

    if (a.requires_grad()) {
        float* grad_a = a.grad().data();
        const float* val_b = b.data(); // Access data directly, assuming contiguous for now or similar layout
        for(int i=0; i<n; ++i) {
            grad_a[i] += go[i] / val_b[i];
        }
    }

    if (b.requires_grad()) {
        float* grad_b = b.grad().data();
        const float* val_a = a.data();
        const float* val_b = b.data();
        for (int i = 0; i < n; ++i) {
            grad_b[i] += -go[i] * val_a[i] / (val_b[i] * val_b[i]);
        }
    }
}

void DivTensorGradFn::cuda_implementation(const Tensor grad_output) {
    Tensor a = parents_[0];
    Tensor b = parents_[1];
    
    float* a_ptr = nullptr;
    if (a.requires_grad()) {
        a_ptr = a.grad().data();
    }

    float* b_ptr = nullptr;
    if (b.requires_grad()) {
        b_ptr = b.grad().data();
    }

    launch_div_grad_accumulate_cuda(a_ptr, b_ptr, a.data(), b.data(), grad_output.data(), a.numel());
}
