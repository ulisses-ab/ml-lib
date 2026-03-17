#include "ReluGradFn.h"
#include <stdexcept>

void launch_relu_backward_cuda(const float* in, const float* grad_out, float* grad_in, int n);

void ReluGradFn::cpu_implementation(const Tensor grad_output) {
    Tensor input = parents_[0];

    if (!input.requires_grad()) return;

    Tensor grad_input = input.grad();

    const int size = input.numel();
    const float* input_data = input.data();
    const float* grad_out_data = grad_output.data();
    float* grad_in_data = grad_input.data();

    for (int i = 0; i < size; ++i) {
        if (input_data[i] > 0.0f) {
            grad_in_data[i] = grad_out_data[i];
        } else {
            grad_in_data[i] = 0.0f;
        }
    }
}

void ReluGradFn::cuda_implementation(const Tensor grad_output) {
    Tensor input = parents_[0];
    if (!input.requires_grad()) return;
    Tensor grad_input = input.grad();

    launch_relu_backward_cuda(input.data(), grad_output.data(), grad_input.data(), input.numel());
}