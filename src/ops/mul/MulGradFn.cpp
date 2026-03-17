#include "MulGradFn.h"
#include <stdexcept>
#include "mul.h"
#include "../add/add.h"

void MulGradFn::cpu_implementation(const Tensor grad_output) {
    float* A = parents_[0].data();
    float* gradA = parents_[0].grad().data();

    
    const float* go = grad_output.data();
    int numel = parents_[0].numel();

    for (int i = 0; i < numel; i++) {
        gradA[i] += go[i] * val_;
    }
}


void launch_mul_grad_cuda(float* grad_input, const float* grad_output, float val, int n);

void MulGradFn::cuda_implementation(const Tensor grad_output) {
    launch_mul_grad_cuda(parents_[0].grad().data(), grad_output.data(), val_, parents_[0].numel());
}