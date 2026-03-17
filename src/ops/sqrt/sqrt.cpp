#include "sqrt.h"
#include "../Kernel.h"
#include "../AutogradOp.h"
#include "SqrtGradFn.h"
#include <cmath>

void launch_sqrt_cuda(const float* x, float* y, int n);

class SqrtKernel : public Kernel {
public:
    SqrtKernel() : Kernel(true) {}

    Shape get_output_shape(const std::vector<Tensor>& inputs) override {
        return inputs[0].shape();
    }

protected:
    void validate(const std::vector<Tensor>& inputs) override {
        if (inputs.size() != 1) {
            throw std::invalid_argument("sqrt: expected 1 input");
        }
    }

    void cpu_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        const float* x = inputs[0].data();
        float* y = output.data();
        int n = output.numel();

        for (int i = 0; i < n; ++i) {
            y[i] = std::sqrt(x[i]);
        }
    }

    void cuda_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        launch_sqrt_cuda(inputs[0].data(), output.data(), output.numel());
    }
};

Tensor sqrt(const Tensor& a) {
    AutogradOp op(std::make_shared<SqrtKernel>(), std::make_shared<SqrtGradFn>());
    return op.forward({a});
}
