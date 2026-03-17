#include "div.h"
#include "DivTensorGradFn.h"
#include "../AutogradOp.h"
#include "../Kernel.h"
#include <stdexcept>

void launch_div_tensor_cuda(const float* a, const float* b, float* c, int n);


class DivTensorKernel : public Kernel {
public:
    DivTensorKernel() : Kernel(true) {}

    Shape get_output_shape(const std::vector<Tensor>& inputs) override {
        return inputs[0].shape();
    }

protected:
    void validate(const std::vector<Tensor>& inputs) override {
        if (inputs.size() != 2) throw std::invalid_argument("mul: expected 2 inputs");
        if (inputs[0].shape() != inputs[1].shape()) throw std::invalid_argument("mul: shapes must match");
    }

    void cpu_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        const int size = output.numel();
        const float* A = inputs[0].data();
        const float* B = inputs[1].data();
        float* C = output.data();

        for (int i = 0; i < size; ++i) {
            C[i] = A[i] / B[i];
        }
    }

    void cuda_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        launch_div_tensor_cuda(inputs[0].data(), inputs[1].data(), output.data(), output.numel());
    }
};


Tensor div(const Tensor& a, const Tensor& b) {
    AutogradOp div_op(std::make_shared<DivTensorKernel>(), std::make_shared<DivTensorGradFn>());
    return div_op.forward({a, b});
}
