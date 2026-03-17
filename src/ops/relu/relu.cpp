#include "relu.h"
#include "ReluGradFn.h"
#include "../Kernel.h"
#include "../AutogradOp.h"
#include <stdexcept>
#include <algorithm>

void launch_relu_cuda(const float* in, float* out, int n);

class ReluKernel : public Kernel {
public:
    Shape get_output_shape(const std::vector<Tensor>& inputs) override {
        return inputs[0].shape();
    }

protected:
    void validate(const std::vector<Tensor>& inputs) override {
        if (inputs.size() != 1) {
            throw std::invalid_argument("relu: must have exactly one input");
        }
    }

    void cpu_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        const Tensor& input = inputs[0];
        const int size = input.numel(); 

        for (int i = 0; i < size; ++i) {
            output.data()[i] = std::max(0.0f, input.data()[i]);
        }
    }

    void cuda_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        const Tensor& input = inputs[0];
        launch_relu_cuda(input.data(), output.data(), input.numel());
    }
};

Tensor relu(const Tensor a) {
    AutogradOp relu_op(std::make_shared<ReluKernel>(), std::make_shared<ReluGradFn>());
    return relu_op.forward({a});
}

void relu_(const Tensor a, Tensor output) {
    ReluKernel relu_kernel;
    return relu_kernel.compute({a}, output);
}