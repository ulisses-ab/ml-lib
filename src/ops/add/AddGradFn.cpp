#include "AddGradFn.h"
#include <stdexcept>

void launch_add_grad_cuda(const float* grad_out, float* grad_in, int n);

void AddGradFn::cpu_implementation(const Tensor grad_output) {
    const int size = grad_output.numel();
    const float* gout = grad_output.data();

    if (parents()[0].requires_grad()) {
        float* grad_a = parents_[0].grad().data();
        for (int i = 0; i < size; ++i) {
            grad_a[i] += gout[i];
        }
    }

    if (parents()[1].requires_grad()) {
        float* grad_b = parents_[1].grad().data();
        for (int i = 0; i < size; ++i) {
            grad_b[i] += gout[i];
        }
    }
}

void AddGradFn::cuda_implementation(const Tensor grad_output) {
    if (parents()[0].requires_grad()) {
        launch_add_grad_cuda(grad_output.data(), parents_[0].grad().data(), grad_output.numel());
    }
    if (parents()[1].requires_grad()) {
        launch_add_grad_cuda(grad_output.data(), parents_[1].grad().data(), grad_output.numel());
    }
}

