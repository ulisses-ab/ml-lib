#include "matmul.h"
#include "MatmulGradFn.h"
#include "../Kernel.h"
#include "../AutogradOp.h"
#include <stdexcept>

void launch_matmul_cuda(const Tensor& a, const Tensor& b, Tensor& c);

class MatmulKernel : public Kernel {
public:
    Shape get_output_shape(const std::vector<Tensor>& inputs) override {
        return Shape({inputs[0].shape()[0], inputs[1].shape()[1]});
    }

protected:
    void validate(const std::vector<Tensor>& inputs) override {
        if (inputs.size() != 2) {
            throw std::invalid_argument("matmul: must have two inputs");
        }

        if (inputs[0].num_dims() != 2 || inputs[1].num_dims() != 2) {
            throw std::invalid_argument("matmul: tensors must be 2D");
        }

        if (inputs[0].shape()[1] != inputs[1].shape()[0]) {
            throw std::invalid_argument("matmul: incompatible shapes");
        }
    }

    void cpu_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        const int M = inputs[0].shape()[0];
        const int K = inputs[0].shape()[1];
        const int N = inputs[1].shape()[1];

        Tensor a = inputs[0];
        Tensor b = inputs[1];
        
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;

                for (int k = 0; k < K; ++k) {
                    sum += a(i, k) * b(k, j);
                }
                    
                output(i, j) = sum;
            }
        }
    }
    
    void cuda_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        launch_matmul_cuda(inputs[0], inputs[1], output);
    }
};

void launch_matmul_cuda(const Tensor& a, const Tensor& b, Tensor& c);

Tensor matmul(const Tensor a, const Tensor b) {    
    AutogradOp matmul_op(std::make_shared<MatmulKernel>(), std::make_shared<MatmulGradFn>());
    return matmul_op.forward({a, b});
}

void matmul_(const Tensor a, const Tensor b, Tensor output) {
    MatmulKernel matmul_kernel;
    return matmul_kernel.compute({a, b}, output);
}