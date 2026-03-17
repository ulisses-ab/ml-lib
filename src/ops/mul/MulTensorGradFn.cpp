#include "MulTensorGradFn.h"
#include <stdexcept>
#include <vector>

void launch_mul_grad_accumulate_cuda(float* grad_in, const float* grad_out, const float* other, int n);

void MulTensorGradFn::cpu_implementation(const Tensor grad_output) {
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
            grad_a[i] += go[i] * val_b[i];
        }
    }

    if (b.requires_grad()) {
        float* grad_b = b.grad().data();
        const float* val_a = a.data();
        for(int i=0; i<n; ++i) {
            grad_b[i] += go[i] * val_a[i];
        }
    }
}

void MulTensorGradFn::cuda_implementation(const Tensor grad_output) {
    Tensor a = parents_[0];
    Tensor b = parents_[1];
    
    if (a.requires_grad()) {
        launch_mul_grad_accumulate_cuda(a.grad().data(), grad_output.data(), b.data(), a.numel());
    }

    if (b.requires_grad()) {
        launch_mul_grad_accumulate_cuda(b.grad().data(), grad_output.data(), a.data(), b.numel());
    }
}
