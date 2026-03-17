#include "LayerNormGradFn.h"
#include "layer_norm.h"
#include <stdexcept>
#include <cmath>
#include <vector>

void launch_layer_norm_grad_cuda(
    const float* grad_out,
    const float* x,
    const float* gamma,
    float* grad_x,
    float* grad_gamma,
    float* grad_beta,
    int N,
    int D,
    float eps
);

void LayerNormGradFn::cpu_implementation(const Tensor grad_output) {
    Tensor x = parents_[0];
    Tensor gamma = parents_[1];
    Tensor beta = parents_[2];

    int N = x.shape()[0];
    int D = x.shape()[1];

    const float* x_data = x.contiguous().data();
    const float* gamma_data = gamma.contiguous().data();
    const float* grad_out = grad_output.data();

    // Gradients for gamma and beta
    if (gamma.requires_grad()) {
        float* grad_gamma = gamma.grad().data();
        // grad_gamma = sum(grad_out * normalized, dim=0)
        for (int i = 0; i < N; ++i) {
            // Recompute mean and variance for this row
            float mean = 0.0f;
            for (int j = 0; j < D; ++j) {
                mean += x_data[i * D + j];
            }
            mean /= static_cast<float>(D);

            float variance = 0.0f;
            for (int j = 0; j < D; ++j) {
                float diff = x_data[i * D + j] - mean;
                variance += diff * diff;
            }
            variance /= static_cast<float>(D);
            float inv_std = 1.0f / std::sqrt(variance + eps_);

            // Accumulate grad_gamma
            for (int j = 0; j < D; ++j) {
                float normalized = (x_data[i * D + j] - mean) * inv_std;
                grad_gamma[j] += grad_out[i * D + j] * normalized;
            }
        }
    }

    if (beta.requires_grad()) {
        float* grad_beta = beta.grad().data();
        // grad_beta = sum(grad_out, dim=0)
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < D; ++j) {
                grad_beta[j] += grad_out[i * D + j];
            }
        }
    }

    // Gradient for x
    if (x.requires_grad()) {
        float* grad_x = x.grad().data();

        for (int i = 0; i < N; ++i) {
            // Recompute mean and variance
            float mean = 0.0f;
            for (int j = 0; j < D; ++j) {
                mean += x_data[i * D + j];
            }
            mean /= static_cast<float>(D);

            float variance = 0.0f;
            for (int j = 0; j < D; ++j) {
                float diff = x_data[i * D + j] - mean;
                variance += diff * diff;
            }
            variance /= static_cast<float>(D);
            float inv_std = 1.0f / std::sqrt(variance + eps_);
            float inv_D = 1.0f / static_cast<float>(D);

            // Compute normalized values
            std::vector<float> normalized(D);
            for (int j = 0; j < D; ++j) {
                normalized[j] = (x_data[i * D + j] - mean) * inv_std;
            }

            // Compute grad_x
            // The gradient computation is:
            // grad_x = (gamma / std) * (grad_out - mean(grad_out) - normalized * mean(grad_out * normalized))
            float sum_grad_out = 0.0f;
            float sum_grad_norm = 0.0f;
            for (int j = 0; j < D; ++j) {
                sum_grad_out += grad_out[i * D + j];
                sum_grad_norm += grad_out[i * D + j] * normalized[j];
            }

            for (int j = 0; j < D; ++j) {
                float term1 = grad_out[i * D + j] * inv_D;
                float term2 = normalized[j] * sum_grad_norm * inv_D;
                grad_x[i * D + j] += gamma_data[j] * inv_std * (grad_out[i * D + j] - term1 - term2);
            }
        }
    }
}

void LayerNormGradFn::cuda_implementation(const Tensor grad_output) {
    Tensor x = parents_[0];
    Tensor gamma = parents_[1];
    Tensor beta = parents_[2];

    int N = x.shape()[0];
    int D = x.shape()[1];

    float* grad_x = x.requires_grad() ? x.grad().data() : nullptr;
    float* grad_gamma = gamma.requires_grad() ? gamma.grad().data() : nullptr;
    float* grad_beta = beta.requires_grad() ? beta.grad().data() : nullptr;

    launch_layer_norm_grad_cuda(
        grad_output.data(),
        x.contiguous().data(),
        gamma.contiguous().data(),
        grad_x,
        grad_gamma,
        grad_beta,
        N,
        D,
        eps_
    );
}

