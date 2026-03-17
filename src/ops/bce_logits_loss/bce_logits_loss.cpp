#include "../AutogradOp.h"
#include "../Kernel.h"
#include "BCELogitsLossGradFn.h"
#include "bce_logits_loss.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>

void launch_bce_logits_loss_cuda(const float* input, const float* target, float* output, int n);

class BCELogitsLossKernel : public Kernel {
public:
    Shape get_output_shape(const std::vector<Tensor>& inputs) override {
        return Shape({1});
    }

protected:
    void validate(const std::vector<Tensor>& inputs) override {
        if (inputs.size() != 2) {
            throw std::invalid_argument("bce_logits_loss: must have two inputs (input, target)");
        }
        if (inputs[0].shape() != inputs[1].shape()) {
             throw std::invalid_argument("bce_logits_loss: input and target must have the same shape");
        }
    }

    void cpu_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        const Tensor& input = inputs[0];
        const Tensor& target = inputs[1];
        
        int n = input.numel();
        
        const float* input_ptr = input.data();
        const float* target_ptr = target.data();
        float* out_ptr = output.data();
        
        float total_loss = 0.0f;

        // Stable binary cross entropy with logits
        // loss = max(x, 0) - x * z + log(1 + exp(-abs(x)))
        for (int i = 0; i < n; ++i) {
            float x = input_ptr[i];
            float z = target_ptr[i];
            
            float max_val = (x > 0) ? x : 0;
            float neg_abs_x = (x > 0) ? -x : x; // -abs(x)
            
            total_loss += max_val - x * z + std::log(1.0f + std::exp(neg_abs_x));
        }
        
        out_ptr[0] = total_loss / static_cast<float>(n);
    }

    void cuda_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        const Tensor& input = inputs[0];
        const Tensor& target = inputs[1];
        int n = input.numel();
        
        launch_bce_logits_loss_cuda(
            input.contiguous().data(), 
            target.contiguous().data(), 
            output.data(), 
            n
        );
    }
};

Tensor bce_logits_loss(const Tensor input, const Tensor target) {
    AutogradOp op(std::make_shared<BCELogitsLossKernel>(), std::make_shared<BCELogitsLossGradFn>());
    return op.forward({input, target});
}
