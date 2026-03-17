#include "add.h"
#include "AddGradFn.h"
#include "../AutogradOp.h"
#include "../Kernel.h"
#include <stdexcept>

void launch_add_cuda(const float* a, const float* b, float* c, int n);

class AddKernel : public Kernel {
public:
    Shape get_output_shape(const std::vector<Tensor>& inputs) override {
        return inputs[0].shape();
    }

protected:
    void validate(const std::vector<Tensor>& inputs) override {
        if (inputs.size() != 2) {
            throw std::invalid_argument("add: must have two inputs");
        }

        if (inputs[0].shape() != inputs[1].shape()) {
            throw std::invalid_argument("add: tensors must have the same shape");
        }
    }

    void cpu_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        const size_t size = output.numel();

        const float* A = inputs[0].contiguous().data();
        const float* B = inputs[1].contiguous().data();
        float* C = output.data();

        for (size_t i = 0; i < size; ++i) {
            C[i] = A[i] + B[i];
        }
    }

    void cuda_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        launch_add_cuda(inputs[0].data(), inputs[1].data(), output.data(), output.numel());
    }
};

Tensor add(const Tensor a, const Tensor b) {
    AutogradOp add_op(std::make_shared<AddKernel>(), std::make_shared<AddGradFn>());
    return add_op.forward({a, b});
}

void add_(const Tensor a, const Tensor b, Tensor output) {
    AddKernel add_kernel;
    return add_kernel.compute({a, b}, output);
} 