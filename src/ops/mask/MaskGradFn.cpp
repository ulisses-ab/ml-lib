#include "MaskGradFn.h"
#include <iostream> 

void launch_mask_grad_cuda(float* grad_a, const float* grad_out, int n);

void MaskGradFn::cpu_implementation(const Tensor grad_output) {
    Tensor a = parents_[0];
    float* grad_a = a.grad().data();
    const float* grad_out = grad_output.data();
    int n = a.shape()[0];

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i >= j) {
                grad_a[i*n + j] += grad_out[i*n + j];
            }
        }
    }
}

void MaskGradFn::cuda_implementation(const Tensor grad_output) {
    launch_mask_grad_cuda(parents_[0].grad().data(), grad_output.data(), parents_[0].shape()[0]);
}