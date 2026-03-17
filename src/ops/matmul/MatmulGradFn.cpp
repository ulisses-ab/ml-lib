#include "MatmulGradFn.h"
#include <stdexcept>
#include "matmul.h"
#include "../add/add.h"
#include <iostream>
#include "../contiguous/contiguous.h"
#include "../transpose/transpose.h"

void MatmulGradFn::cpu_implementation(const Tensor grad_output) {
    Tensor A = parents_[0];
    Tensor B = parents_[1];
    Tensor go = grad_output;

    if (go.num_dims() != 2) {
        go = go.view(Shape({A.shape()[0], B.shape()[1]}));
    }

    if (A.requires_grad()) {
        Tensor gradA = A.grad();
        Tensor Bt = transpose(B, 0, 1).detach().contiguous();

        Tensor h(gradA.shape(), A.device());
        matmul_(contiguous_(go), Bt, h);
        add_(gradA, h, gradA);
    }

    if (B.requires_grad()) {
        Tensor gradB = B.grad();
        Tensor At = transpose(A, 0, 1).detach().contiguous();
    
        Tensor h(gradB.shape(), A.device());
        matmul_(At, contiguous_(go), h);
        add_(gradB, h, gradB);
    }
}

void MatmulGradFn::cuda_implementation(const Tensor grad_output) {
    cpu_implementation(grad_output);    
}