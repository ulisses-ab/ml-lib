#include "concat.h"
#include "ConcatGradFn.h"
#include "../AutogradOp.h"
#include "../Kernel.h"

#include <stdexcept>
#include <numeric>
#include <cstring>
#include <vector>

void launch_concat_cuda(
    const std::vector<const float*>& input_ptrs,
    float* output,
    const std::vector<int>& input_cols,
    int rows,
    int out_cols,
    int tail
);

class ConcatKernel : public Kernel {
public:
    Shape get_output_shape(const std::vector<Tensor>& inputs) override {
        Shape out = inputs[0].shape();
        out[1] = 0;

        for (const auto& t : inputs) {
            out[1] += t.shape()[1];
        }

        return out;
    }

protected:
    void validate(const std::vector<Tensor>& inputs) override {
        if (inputs.empty()) {
            throw std::invalid_argument("concat: must have at least one input");
        }

        const Shape& base = inputs[0].shape();

        for (const auto& t : inputs) {
            if (t.num_dims() != base.num_dims()) {
                throw std::invalid_argument("concat: all tensors must have same rank");
            }

            for (size_t d = 0; d < base.num_dims(); ++d) {
                if (d == 1) continue; // allow mismatch on dim 1
                if (t.shape()[d] != base[d]) {
                    throw std::invalid_argument(
                        "concat: all tensors must match in all dims except dim 1"
                    );
                }
            }
        }
    }

    void cpu_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        float* out_ptr = output.data();

        size_t rows = output.shape()[0];
        size_t out_cols = output.shape()[1];

        // Product of dims after dim 1
        size_t tail = 1;
        for (size_t d = 2; d < output.num_dims(); ++d) {
            tail *= output.shape()[d];
        }

        size_t col_offset = 0;

        for (const auto& t : inputs) {
            Tensor tc = t.contiguous();

            size_t cols = tc.shape()[1];
            const float* in_ptr = tc.data();

            for (size_t r = 0; r < rows; ++r) {
                std::memcpy(
                    out_ptr + (r * out_cols + col_offset) * tail,
                    in_ptr  + (r * cols) * tail,
                    cols * tail * sizeof(float)
                );
            }

            col_offset += cols;
        }
    }

    void cuda_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        size_t rows = output.shape()[0];
        size_t out_cols = output.shape()[1];
        
        // Product of dims after dim 1
        size_t tail = 1;
        for (size_t d = 2; d < output.num_dims(); ++d) {
            tail *= output.shape()[d];
        }
        
        // Prepare input pointers and column sizes
        std::vector<const float*> input_ptrs;
        std::vector<int> input_cols;
        
        for (const auto& t : inputs) {
            Tensor tc = t.contiguous();
            input_ptrs.push_back(tc.data());
            input_cols.push_back(tc.shape()[1]);
        }
        
        launch_concat_cuda(input_ptrs, output.data(), input_cols, rows, out_cols, tail);
    }
};

Tensor concat(const std::vector<Tensor>& inputs) {
    AutogradOp op(
        std::make_shared<ConcatKernel>(),
        std::make_shared<ConcatGradFn>()
    );
    return op.forward(inputs);
}
