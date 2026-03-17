#include "mask.h"
#include "../../core/Tensor.h"
#include <stdexcept>
#include "../fill/fill.h"
#include "../Kernel.h"
#include "../AutogradOp.h"
#include "MaskGradFn.h"

void launch_mask_cuda(const float* a, float* b, int n);

class MaskKernel : public Kernel {
public:
    Shape get_output_shape(const std::vector<Tensor>& inputs) override {
        return inputs[0].shape();
    }

protected:
    void validate(const std::vector<Tensor>& inputs) override {
        if (inputs.size() != 1) {
            throw std::invalid_argument("mask: must have one input");
        }

        if (inputs[0].num_dims() != 2) {
            throw std::invalid_argument("mask: input must be a 2D tensor");
        }

        if (inputs[0].shape()[0] != inputs[0].shape()[1]) {
            throw std::invalid_argument("mask: input must be a square matrix");
        }
    }

    void cpu_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        const size_t size = output.numel();

        const float* A = inputs[0].contiguous().data();
        float* B = output.data();
        int n = inputs[0].shape()[0];

        fill_(output, -1e9);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i >= j) {
                    B[i * n + j] = A[i * n + j];
                }
            }
        }
    }

    void cuda_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        launch_mask_cuda(inputs[0].data(), output.data(), output.shape()[0]);
    }
};

Tensor mask(const Tensor a) {
    AutogradOp mask_op(std::make_shared<MaskKernel>(), std::make_shared<MaskGradFn>());
    return mask_op.forward({a});
}