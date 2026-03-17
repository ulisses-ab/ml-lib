#include "SqrtGradFn.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

void launch_sqrt_grad_cuda(float* grad_x, const float* grad_y, const float* x, int n);

void SqrtGradFn::cpu_implementation(const Tensor grad_output) {
    if (parents_.size() != 1) return;
    Tensor input = parents_[0];
    
    if (input.requires_grad()) {
        float* grad_x = input.grad().data();
        const float* grad_y = grad_output.data();
        const float* x = input.data();
        int n = input.numel();

        for (int i = 0; i < n; ++i) {
            if (x[i] > 1e-12f) {
                grad_x[i] += grad_y[i] * (0.5f / std::sqrt(x[i]));
            }
        }
    }
}

void SqrtGradFn::cuda_implementation(const Tensor grad_output) {
    if (parents_.size() != 1) return;
    Tensor input = parents_[0];

    if (input.requires_grad()) {
        launch_sqrt_grad_cuda(input.grad().data(), grad_output.data(), input.data(), input.numel());
    }
}
