#include "SoftmaxGradFn.h"
#include <cmath>
#include <vector>
#include <vector>
#include <stdexcept>

void launch_softmax_grad_cuda(const float* grad_output, const float* input, float* grad_input, int rows, int cols);

void SoftmaxGradFn::cpu_implementation(const Tensor grad_output) {
    Tensor input = parents_[0];
    Tensor grad_input = input.grad();

    const int rows = input.shape()[0];
    const int cols = input.shape()[1];

    const float* x = input.data();
    const float* g = grad_output.data();
    float* dx = grad_input.data();

    // buffer temporário para o softmax de uma linha
    std::vector<float> y(cols);

    for (int r = 0; r < rows; ++r) {
        const float* x_row = x + r * cols;
        const float* g_row = g + r * cols;
        float* dx_row = dx + r * cols;

        // 1. softmax forward (estável)
        float max_val = x_row[0];
        for (int c = 1; c < cols; ++c) {
            if (x_row[c] > max_val) {
                max_val = x_row[c];
            }
        }

        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            y[c] = std::exp(x_row[c] - max_val);
            sum += y[c];
        }

        for (int c = 0; c < cols; ++c) {
            y[c] /= sum;
        }

        // 2. dot = sum_j (g_j * y_j)
        float dot = 0.0f;
        for (int c = 0; c < cols; ++c) {
            dot += g_row[c] * y[c];
        }

        // 3. grad_input
        for (int c = 0; c < cols; ++c) {
            dx_row[c] += y[c] * (g_row[c] - dot);
        }
    }
}

void SoftmaxGradFn::cuda_implementation(const Tensor grad_output) {
    Tensor input = parents_[0];
    Tensor grad_input = input.grad();

    launch_softmax_grad_cuda(grad_output.data(), input.data(), grad_input.data(), input.shape()[0], input.shape()[1]);
}
