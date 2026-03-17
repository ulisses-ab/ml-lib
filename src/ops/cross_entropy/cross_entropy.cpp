#include "../AutogradOp.h"
#include "../Kernel.h"
#include "CrossEntropyGradFn.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>

class CrossEntropyKernel : public Kernel {
public:
    // Output is a scalar (mean loss)
    Shape get_output_shape(const std::vector<Tensor>& inputs) override {
        return Shape({1}); 
    }

protected:
    void validate(const std::vector<Tensor>& inputs) override {
        if (inputs.size() != 2) {
            throw std::invalid_argument("crossentropy: must have two inputs (logits, targets)");
        }
        // logits: [Batch, Classes], targets: [Batch]
        if (inputs[0].shape()[0] != inputs[1].numel()) {
            throw std::invalid_argument("crossentropy: batch size mismatch");
        }
    }

    void cpu_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        const Tensor& logits = inputs[0];
        const Tensor& targets = inputs[1];
        
        const int batch_size = logits.shape()[0];
        const int num_classes = logits.shape()[1];
        
        const float* logits_ptr = logits.data();
        const float* targets_ptr = targets.data();
        float* out_data = output.data();

        float total_loss = 0.0f;

        for (int b = 0; b < batch_size; ++b) {
            const float* row = logits_ptr + (b * num_classes);
            int target_idx = static_cast<int>(targets_ptr[b]);

            // Find max
            float max_logit = row[0];
            for (int c = 1; c < num_classes; ++c) {
                max_logit = std::max(max_logit, row[c]);
            }

            // Compute Log-Sum-Exp
            float sum_exp = 0.0f;
            for (int c = 0; c < num_classes; ++c) {
                sum_exp += std::exp(row[c] - max_logit);
            }
            float log_sum_exp = max_logit + std::log(sum_exp);

            // Loss = -logits[target] + log_sum_exp
            total_loss += (log_sum_exp - row[target_idx]);
        }

        out_data[0] = total_loss / static_cast<float>(batch_size);
    }

    void cuda_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        throw std::logic_error("crossentropy: cuda not implemented");
    }
};

Tensor cross_entropy(const Tensor logits, const Tensor targets) {
    AutogradOp ce_op(std::make_shared<CrossEntropyKernel>(), std::make_shared<CrossEntropyGradFn>());
    return ce_op.forward({logits, targets});
}
