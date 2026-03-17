#include "contiguous.h"
#include "../Kernel.h"
#include "../AutogradOp.h"
#include <stdexcept>
#include "ContiguousGradFn.h"

void launch_contiguous_2d_cuda(
    const float* src, 
    float* dst, 
    int M,
    int N,
    int row_stride,
    int col_string
);

class ContiguousKernel : public Kernel {
public:
    ContiguousKernel() : Kernel(false) {}

    Shape get_output_shape(const std::vector<Tensor>& inputs) override {
        return inputs[0].shape();
    }

protected:
    void validate(const std::vector<Tensor>& inputs) override {
        if (inputs.size() != 1) {
            throw std::invalid_argument("contiguous: must have exactly one input");
        }
    }

    void cpu_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        const Tensor input = inputs[0];
        const Shape& shape = output.shape();
        
        int total_elements = shape.numel(); 
        int rank = shape.num_dims();

        std::vector<int> indices(rank);

        for (int i = 0; i < total_elements; ++i) {
            int temp_index = i;
            
            for (int d = rank - 1; d >= 0; --d) {
                indices[d] = temp_index % shape[d];
                temp_index /= shape[d];
            }

            const float value = input.at(indices);

            output.data()[i] = value;
        }
    }

    void cuda_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        Tensor src = inputs[0];

        if (src.num_dims() == 2) {
            launch_contiguous_2d_cuda(
                src.raw_data() + src.layout().base_offset(), 
                output.data(), 
                src.shape()[0], 
                src.shape()[1],
                src.layout().stride(0),
                src.layout().stride(1)
            );
            return;
        }

        throw std::logic_error("contiguous: cuda not implemented for this number of dimensions");
    }
};

Tensor contiguous(const Tensor a) {
    AutogradOp cont_op(std::make_shared<ContiguousKernel>(), std::make_shared<ContiguousGradFn>());
    return cont_op.forward({a});
}

Tensor contiguous_(const Tensor a) {
    Tensor output(a.shape(), a.device(), a.requires_grad());
    ContiguousKernel contiguous_kernel;
    contiguous_kernel.compute({a}, output);
    return output;
}