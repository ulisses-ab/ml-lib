#include "mse.h"
#include "MSEGradFn.h"
#include "../AutogradOp.h"
#include "../Kernel.h"
#include <stdexcept>
#include <cmath>

class MSEKernel : public Kernel {
public:
    Shape get_output_shape(const std::vector<Tensor>& inputs) override {
        return Shape({1}); 
    }

protected:
    void validate(const std::vector<Tensor>& inputs) override {
        if (inputs.size() != 2) {
            throw std::invalid_argument("mse: must have two inputs (pred, target)");
        }
        if (inputs[0].shape() != inputs[1].shape()) {
            throw std::invalid_argument("mse: inputs must have the same shape");
        }
    }

    void cpu_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        const int n = inputs[0].numel();
        const float* pred = inputs[0].data();
        const float* target = inputs[1].data();
        float* out_data = output.data();

        float sum_sq_diff = 0.0f;
        for (int i = 0; i < n; ++i) {
            float diff = pred[i] - target[i];
            sum_sq_diff += diff * diff;
        }

        out_data[0] = sum_sq_diff / static_cast<float>(n);
    }

    void cuda_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        throw std::logic_error("mse: cuda not implemented");
    }
};

Tensor mse(const Tensor pred, const Tensor target) {
    AutogradOp mse_op(std::make_shared<MSEKernel>(), std::make_shared<MSEGradFn>());
    return mse_op.forward({pred, target});
}

void mse_(const Tensor pred, const Tensor target, Tensor output) {
    MSEKernel mse_kernel;
    return mse_kernel.compute({pred, target}, output);
}