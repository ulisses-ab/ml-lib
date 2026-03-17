#include "softmax.h"
#include "SoftmaxGradFn.h"
#include "../AutogradOp.h"
#include "../Kernel.h"
#include <stdexcept>
#include <cmath>

void launch_softmax_cuda(const float* A, float* B, int rows, int cols);

class SoftmaxKernel : public Kernel {
public:
    Shape get_output_shape(const std::vector<Tensor>& inputs) override {
        return inputs[0].shape();
    }

protected:
    void validate(const std::vector<Tensor>& inputs) override {
        if (inputs.size() != 1) {
            throw std::invalid_argument("must have 1 input");
        }

        if (inputs[0].num_dims() != 2) {
            throw std::invalid_argument("input must be 2D");
        } 
    }

    void cpu_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        const Tensor& A = inputs[0];

        const int rows = A.shape()[0];
        const int cols = A.shape()[1];

        const float* A_data = A.data();
        float* B_data = output.data();

        for (int r = 0; r < rows; ++r) {
            float max_val = A_data[r * cols];
            for (int c = 1; c < cols; ++c) {
                float v = A_data[r * cols + c];
                if (v > max_val) {
                    max_val = v;
                }
            }

            float sum = 0.0f;
            for (int c = 0; c < cols; ++c) {
                float e = std::exp(A_data[r * cols + c] - max_val);
                B_data[r * cols + c] = e;
                sum += e;
            }

            for (int c = 0; c < cols; ++c) {
                B_data[r * cols + c] /= sum;
            }
        }
    }


    void cuda_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        launch_softmax_cuda(inputs[0].data(), output.data(), inputs[0].shape()[0], inputs[0].shape()[1]);
    }
};

Tensor softmax(const Tensor a) {
    AutogradOp softmax_op(std::make_shared<SoftmaxKernel>(), std::make_shared<SoftmaxGradFn>());
    return softmax_op.forward({a});
}

void softmax_(const Tensor a, Tensor b) {
    SoftmaxKernel mul_kernel;
    return mul_kernel.compute({a}, b);
}