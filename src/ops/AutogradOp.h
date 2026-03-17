#pragma once

#include "../core/Tensor.h"
#include <vector>
#include <stdexcept>
#include "Kernel.h"
#include "../grad/GradFn.h"

class AutogradOp {
public:
    AutogradOp(std::shared_ptr<Kernel> kernel, std::shared_ptr<GradFn> grad_fn) : kernel_(kernel), grad_fn_(grad_fn) {}

    Tensor forward(const std::vector<Tensor>& inputs) {
        Shape shape = kernel_->get_output_shape(inputs);
        Tensor output = create_output_tensor(shape, inputs.at(0).device(), inputs);

        kernel_->compute(inputs, output);

        return output;
    }

private:
    Tensor create_output_tensor(
        const Shape& shape, 
        Device device, 
        const std::vector<Tensor>& inputs
    ) {
        bool requires_grad = false;

        for (const auto& input : inputs) {
            if (input.requires_grad()) requires_grad = true;
        }

        std::shared_ptr<GradFn> grad_fn = nullptr;

        if (requires_grad) {
            grad_fn_->add_parents(inputs);
            grad_fn = grad_fn_;
        }

        return Tensor(shape, device, requires_grad, grad_fn);
    }

    std::shared_ptr<Kernel> kernel_;
    std::shared_ptr<GradFn> grad_fn_;
};