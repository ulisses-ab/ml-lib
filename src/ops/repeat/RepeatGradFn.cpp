#include "RepeatGradFn.h"
#include <stdexcept>

void RepeatGradFn::cpu_implementation(const Tensor grad_output) {
    Tensor input = parents_[0];
    Tensor grad_input = input.grad();

    const int rows = input.shape()[0];
    const int repeats = grad_output.shape()[0]; 

    const float* grad_out_data = grad_output.data();
    float* grad_in_data = grad_input.data();

    for (int r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (int c = 0; c < repeats; ++c) {
            sum += grad_out_data[c * rows + r];
        }
        grad_in_data[r] += sum;
    }
}

void launch_repeat_grad_cuda(
    const float* grad_out,
    float* grad_in,
    int rows,
    int repeats
);

void RepeatGradFn::cuda_implementation(const Tensor grad_output) {
    Tensor input = parents_[0];
    Tensor grad_input = input.grad();

    const int rows = input.shape()[0];
    const int repeats = grad_output.shape()[0];

    launch_repeat_grad_cuda(
        grad_output.data(),
        grad_input.data(),
        rows,
        repeats
    );
}
