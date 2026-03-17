#include "transpose.h"
#include "../Kernel.h"
#include "../AutogradOp.h"
#include "TransposeGradFn.h"
#include <algorithm>
#include <vector>

void launch_transpose_2d_cuda(const float* src, float* dst, int rows, int cols);

class TransposeKernel : public Kernel {
public:
    TransposeKernel(int dim0, int dim1) : Kernel(true), dim0_(dim0), dim1_(dim1) {}

    Shape get_output_shape(const std::vector<Tensor>& inputs) override {
        Shape s = inputs[0].shape();
        int r = s.num_dims();
        if (dim0_ < 0 || dim0_ >= r || dim1_ < 0 || dim1_ >= r) {
             throw std::invalid_argument("transpose: dimension out of range");
        }
        int t = s[dim0_];
        s[dim0_] = s[dim1_];
        s[dim1_] = t;
        return s;
    }

protected:
    void validate(const std::vector<Tensor>& inputs) override {
        if (inputs.size() != 1) {
            throw std::invalid_argument("transpose: expected 1 input");
        }
    }

    void cpu_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        const Tensor& input = inputs[0];
        const Shape& out_shape = output.shape();
        int rank = out_shape.num_dims();
        int numel = out_shape.numel();
        std::vector<int> out_idx(rank);
        
        // This is slow but generic
        for (int i=0; i < numel; ++i) {
             int t = i;
             for (int d = rank - 1; d >= 0; --d) {
                 out_idx[d] = t % out_shape[d];
                 t /= out_shape[d];
             }
             
             std::vector<int> in_idx = out_idx;
             // We are conceptually transposing: output[i, j] = input[j, i]
             // If we iterate output indices (that correspond to physical layout if contiguous),
             // we need to source from swapped indices.
             int temp = in_idx[dim0_];
             in_idx[dim0_] = in_idx[dim1_];
             in_idx[dim1_] = temp;
             
             output.at(out_idx) = input.at(in_idx);
        }
    }

    void cuda_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        const Tensor& input = inputs[0];
        int rank = input.num_dims();

        if (rank == 2) {
            // If dims are 0 and 1, it's a full transpose.
            // If dims are same, it's a copy.
            if (dim0_ == dim1_) {
                // Should use copy? But logic handles it (swap is no-op).
                // Just use launch_transpose. 
            }
            // For 2D, dims must be 0 and 1.
            // Basic sanity check
            launch_transpose_2d_cuda(
                input.data(),
                output.data(),
                input.shape()[0],
                input.shape()[1]
            );
            return;
        }

        throw std::logic_error("transpose: cuda not implemented for N-d tensors (only 2D supported for now)");
    }

private:
    int dim0_;
    int dim1_;
};

Tensor transpose(const Tensor& a, int dim0, int dim1) {
    AutogradOp op(std::make_shared<TransposeKernel>(dim0, dim1), std::make_shared<TransposeGradFn>(dim0, dim1));
    return op.forward({a});
}
