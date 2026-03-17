#include "gather.h"
#include "GatherGradFn.h"
#include "../AutogradOp.h"
#include "../Kernel.h"
#include <stdexcept>
#include <cmath>

void launch_gather_cuda(const int* indices, const float* embedding, float* output, int N, int D, int V);

class GatherKernel : public Kernel {
public:
    Shape get_output_shape(const std::vector<Tensor>& inputs) override {
        const Tensor& indices = inputs[0];
        const Tensor& embedding = inputs[1];
        
        int N = indices.numel();
        int D = embedding.shape()[1];
        
        return Shape({N, D});
    }

protected:
    void validate(const std::vector<Tensor>& inputs) override {
        if (inputs.size() != 2) {
            throw std::invalid_argument("gather: must have two inputs (indices, embedding)");
        }

        const Tensor& indices = inputs[0];
        const Tensor& embedding = inputs[1];

        if (indices.num_dims() != 1) {
            throw std::invalid_argument("gather: indices must be 1D");
        }

        if (embedding.num_dims() != 2) {
            throw std::invalid_argument("gather: embedding must be 2D");
        }
    }

    void cpu_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        const Tensor& indices = inputs[0];
        const Tensor& embedding = inputs[1];
        
        int N = indices.numel();
        int D = embedding.shape()[1];
        int V = embedding.shape()[0];
        
        const float* indices_data = indices.contiguous().data();
        const float* embedding_data = embedding.contiguous().data();
        float* output_data = output.data();
        
        for (int i = 0; i < N; ++i) {
            int idx = static_cast<int>(std::round(indices_data[i]));
            
            if (idx < 0 || idx >= V) {
                throw std::out_of_range("gather: index out of range");
            }
            
            // Copy the idx-th row from embedding to output[i]
            for (int j = 0; j < D; ++j) {
                output_data[i * D + j] = embedding_data[idx * D + j];
            }
        }
    }

    void cuda_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        const Tensor& indices = inputs[0];
        const Tensor& embedding = inputs[1];
        
        int N = indices.numel();
        int D = embedding.shape()[1];
        int V = embedding.shape()[0];
        
        // Cast indices to int* for CUDA kernel
        const int* indices_data = reinterpret_cast<const int*>(indices.contiguous().data());
        const float* embedding_data = embedding.contiguous().data();
        float* output_data = output.data();
        
        launch_gather_cuda(indices_data, embedding_data, output_data, N, D, V);
    }
};

Tensor gather(const Tensor indices, const Tensor embedding) {
    AutogradOp gather_op(std::make_shared<GatherKernel>(), std::make_shared<GatherGradFn>());
    return gather_op.forward({indices, embedding});
}

void gather_(const Tensor indices, const Tensor embedding, Tensor output) {
    GatherKernel gather_kernel;
    return gather_kernel.compute({indices, embedding}, output);
}

