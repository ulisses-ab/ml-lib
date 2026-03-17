#include "CrossEntropyGradFn.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <iostream>

void CrossEntropyGradFn::cpu_implementation(const Tensor grad_output) {
    // inputs_[0] is logits, inputs_[1] is targets
    Tensor logits = parents_[0];
    const Tensor targets = parents_[1];

    if (!logits.requires_grad()) return;

    Tensor grad_logits = logits.grad();

    const int batch_size = logits.shape()[0];
    const int num_classes = logits.shape()[1];
    
    const float* logits_ptr = logits.data();
    const float* targets_ptr = targets.data();
    float* grad_ptr = grad_logits.data();
    
    // The gradient flowing from the loss (usually a scalar 1.0)
    float upstream_grad = grad_output.data()[0];

    for (int b = 0; b < batch_size; ++b) {
        const float* row_logits = logits_ptr + (b * num_classes);
        float* row_grad = grad_ptr + (b * num_classes);
        int target_idx = static_cast<int>(targets_ptr[b]);

        // 1. Compute Softmax for this row (re-using stable log-sum-exp logic)
        float max_logit = row_logits[0];
        for (int c = 1; c < num_classes; ++c) {
            max_logit = std::max(max_logit, row_logits[c]);
        }

        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
            sum_exp += std::exp(row_logits[c] - max_logit);
        }

        // 2. Compute gradient: (Softmax(z_i) - Target_i) * upstream / batch_size
        for (int c = 0; c < num_classes; ++c) {
            float softmax_p = std::exp(row_logits[c] - max_logit) / sum_exp;
            float target_val = (c == target_idx) ? 1.0f : 0.0f;
            
            row_grad[c] += (softmax_p - target_val) * upstream_grad / static_cast<float>(batch_size);
        }
    }
}

void CrossEntropyGradFn::cuda_implementation(const Tensor grad_output) {
    throw std::logic_error("CrossEntropy backward: CUDA not implemented");
}