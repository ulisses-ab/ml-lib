#include "layer_norm.h"
#include "LayerNormGradFn.h"
#include "../AutogradOp.h"
#include "../Kernel.h"
#include <stdexcept>
#include <cmath>

void launch_layer_norm_cuda(const float* x, const float* gamma, const float* beta, float* output, int N, int D, float eps);

class LayerNormKernel : public Kernel {
public:
    LayerNormKernel(float eps) : eps_(eps) {}

    Shape get_output_shape(const std::vector<Tensor>& inputs) override {
        return inputs[0].shape();
    }

protected:
    void validate(const std::vector<Tensor>& inputs) override {
        if (inputs.size() != 3) {
            throw std::invalid_argument("layer_norm: must have three inputs (x, gamma, beta)");
        }

        const Tensor& x = inputs[0];
        const Tensor& gamma = inputs[1];
        const Tensor& beta = inputs[2];

        if (x.num_dims() != 2) {
            throw std::invalid_argument("layer_norm: input x must be 2D");
        }

        int d_model = x.shape()[1];

        if (gamma.num_dims() != 1 || gamma.shape()[0] != d_model) {
            throw std::invalid_argument("layer_norm: gamma must be 1D with size matching last dimension of x");
        }

        if (beta.num_dims() != 1 || beta.shape()[0] != d_model) {
            throw std::invalid_argument("layer_norm: beta must be 1D with size matching last dimension of x");
        }
    }

    void cpu_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        const Tensor& x = inputs[0];
        const Tensor& gamma = inputs[1];
        const Tensor& beta = inputs[2];

        int N = x.shape()[0];  // batch_size * seq_len
        int D = x.shape()[1];   // d_model

        const float* x_data = x.contiguous().data();
        const float* gamma_data = gamma.contiguous().data();
        const float* beta_data = beta.contiguous().data();
        float* out_data = output.data();

        for (int i = 0; i < N; ++i) {
            // Compute mean
            float mean = 0.0f;
            for (int j = 0; j < D; ++j) {
                mean += x_data[i * D + j];
            }
            mean /= static_cast<float>(D);

            // Compute variance
            float variance = 0.0f;
            for (int j = 0; j < D; ++j) {
                float diff = x_data[i * D + j] - mean;
                variance += diff * diff;
            }
            variance /= static_cast<float>(D);

            // Normalize and apply affine transformation
            float inv_std = 1.0f / std::sqrt(variance + eps_);
            for (int j = 0; j < D; ++j) {
                float normalized = (x_data[i * D + j] - mean) * inv_std;
                out_data[i * D + j] = gamma_data[j] * normalized + beta_data[j];
            }
        }
    }

    void cuda_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        const Tensor& x = inputs[0];
        const Tensor& gamma = inputs[1];
        const Tensor& beta = inputs[2];

        int N = x.shape()[0];
        int D = x.shape()[1];

        launch_layer_norm_cuda(
            x.contiguous().data(),
            gamma.contiguous().data(),
            beta.contiguous().data(),
            output.data(),
            N,
            D,
            eps_
        );
    }

private:
    float eps_;
};

Tensor layer_norm(const Tensor x, const Tensor gamma, const Tensor beta, float eps) {
    AutogradOp ln_op(std::make_shared<LayerNormKernel>(eps), std::make_shared<LayerNormGradFn>(eps));
    return ln_op.forward({x, gamma, beta});
}

void layer_norm_(const Tensor x, const Tensor gamma, const Tensor beta, Tensor output, float eps) {
    LayerNormKernel ln_kernel(eps);
    return ln_kernel.compute({x, gamma, beta}, output);
}


