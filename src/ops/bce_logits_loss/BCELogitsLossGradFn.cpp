#include "BCELogitsLossGradFn.h"
#include <cmath>
#include <vector>

void launch_bce_logits_loss_grad_cuda(const float* input, const float* target, const float* grad_output, float* grad_input, int n);

void BCELogitsLossGradFn::cpu_implementation(const Tensor grad_output) {
    Tensor input = parents_[0];
    Tensor target = parents_[1];

    if (!input.requires_grad()) return;

    Tensor grad_input = input.grad();
    int n = input.numel();

    const float* input_ptr = input.data();
    const float* target_ptr = target.data();
    const float upstream_grad = grad_output.data()[0];
    float* grad_ptr = grad_input.data();
    
    for (int i = 0; i < n; ++i) {
        float x = input_ptr[i];
        float z = target_ptr[i];
        
        // dL/dx = sigmoid(x) - z
        // sigmoid(x) = 1 / (1 + exp(-x))
        float sigmoid_x = 1.0f / (1.0f + std::exp(-x));
        
        float grad = sigmoid_x - z;
        grad_ptr[i] += grad * upstream_grad / static_cast<float>(n);
    }
}

void BCELogitsLossGradFn::cuda_implementation(const Tensor grad_output) {
    Tensor input = parents_[0];
    Tensor target = parents_[1];

    if (!input.requires_grad()) return;

    int n = input.numel();
    
    launch_bce_logits_loss_grad_cuda(
        input.contiguous().data(),
        target.contiguous().data(),
        grad_output.data(),
        input.grad().data(),
        n
    );
}
