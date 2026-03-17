#include "repeat.h"
#include "RepeatGradFn.h"
#include "../AutogradOp.h"
#include "../Kernel.h"
#include <stdexcept>
#include <memory>

void launch_repeat_cuda(const float* src, float* dst, int src_size, int repeats);

class RepeatKernel : public Kernel {
public:
    RepeatKernel(int val) : val_(val) {}

    Shape get_output_shape(const std::vector<Tensor>& inputs) override {
        Shape in_shape = inputs[0].shape();
        
        return Shape({val_, in_shape[0]});
    }

protected:
    void validate(const std::vector<Tensor>& inputs) override {
        if (inputs.size() != 1) {
            throw std::invalid_argument("repeat: needs input tensor and repeat count scalar");
        }

        if (inputs[0].num_dims() != 1) {
            throw std::invalid_argument("repeat: first input must be 1D");
        }
    }

    void cpu_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        Tensor input = inputs[0];
        const int rows = input.shape()[0];
        const int repeats = val_;

        const float* src = input.data();
        float* dst = output.data();

        for (int r = 0; r < rows; ++r) {
            float val = src[r];
            for (int c = 0; c < repeats; ++c) {
                dst[c * rows + r] = val;
            }
        }
    }

    void cuda_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        launch_repeat_cuda(inputs[0].data(), output.data(), inputs[0].shape()[0], val_);
    }

private:
    int val_;
};

Tensor repeat(int val, const Tensor a) {
    AutogradOp repeat_op(std::make_shared<RepeatKernel>(val), std::make_shared<RepeatGradFn>());
    return repeat_op.forward({a});
}

void repeat_(int val, const Tensor a, Tensor b) {
    RepeatKernel repeat_kernel(val);
    return repeat_kernel.compute({a}, b);
}