#include "mul.h"
#include "MulGradFn.h"
#include "MulTensorGradFn.h"
#include "../AutogradOp.h"
#include "../Kernel.h"
#include <stdexcept>

void launch_mul_cuda(const float* A, float* B, float val, int n);
void launch_mul_tensor_cuda(const float* a, const float* b, float* c, int n);

class MulKernel : public Kernel {
public:
    MulKernel(float val) : val_(val) {}

    Shape get_output_shape(const std::vector<Tensor>& inputs) override {
        return inputs[0].shape();
    }

protected:
    void validate(const std::vector<Tensor>& inputs) override {
    }

    void cpu_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        const int size = output.numel();

        const float* A = inputs[0].data();
        float* C = output.data();

        for (int i = 0; i < size; ++i) {
            C[i] = A[i] * val_;
        }
    }

    void cuda_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        launch_mul_cuda(inputs[0].data(), output.data(),  val_, inputs[0].numel());
    }

private:
    float val_;
};

class MulTensorKernel : public Kernel {
public:
    MulTensorKernel() : Kernel(true) {}

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
            C[i] = A[i] * B[i];
        }
    }

    void cuda_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        launch_mul_tensor_cuda(inputs[0].data(), inputs[1].data(), output.data(), output.numel());
    }
};

Tensor mul(float val, const Tensor a) {
    AutogradOp mul_op(std::make_shared<MulKernel>(val), std::make_shared<MulGradFn>(val));
    return mul_op.forward({a});
}

Tensor mul(const Tensor& a, const Tensor& b) {
    AutogradOp mul_op(std::make_shared<MulTensorKernel>(), std::make_shared<MulTensorGradFn>());
    return mul_op.forward({a, b});
}

void mul_(float val, const Tensor a, Tensor b) {
    MulKernel mul_kernel(val);
    return mul_kernel.compute({a}, b);
}