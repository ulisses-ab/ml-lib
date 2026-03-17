#include "CopyGradFn.h"
#include "../add/add.h"
#include "copy.h"

void CopyGradFn::backward(const Tensor grad_output) {
    auto& parent = parents_[0];

    if (parent.requires_grad()) {
        Tensor parent_grad = parent.grad();
        
        if (grad_output.device() != parent.device()) {
            Tensor local_grad = copy(grad_output, parent.device());
            add_(parent_grad, local_grad, parent_grad);
        } else {
            add_(parent_grad, grad_output, parent_grad);
        }
    }
}
