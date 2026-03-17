#include "MSEGradFn.h"
#include <stdexcept>

void MSEGradFn::cpu_implementation(const Tensor grad_output) {
    const float gout = grad_output.data()[0];

    Tensor pred_tensor = parents_[0];
    Tensor target_tensor = parents_[1];
    
    const int n = pred_tensor.numel();
    const float inv_n = 1.0f / static_cast<float>(n);
    const float scale = 2.0f * inv_n * gout;

    const float* pred = pred_tensor.data();
    const float* target = target_tensor.data();

    if (pred_tensor.requires_grad()) {
        float* grad_pred = pred_tensor.grad().data();
        for (int i = 0; i < n; ++i) {
            grad_pred[i] += scale * (pred[i] - target[i]);
        }
    }

    if (target_tensor.requires_grad()) {
        float* grad_target = target_tensor.grad().data();
        for (int i = 0; i < n; ++i) {
            grad_target[i] -= scale * (pred[i] - target[i]);
        }
    }
}

void MSEGradFn::cuda_implementation(const Tensor grad_output) {
    throw std::logic_error("mse grad fn: cuda not implemented");
}