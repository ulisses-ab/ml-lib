#pragma once

#include "../../grad/GradFn.h"

class CopyGradFn : public GradFn {
public:
    using GradFn::GradFn;

    void backward(const Tensor grad_output) override;
    std::string name() override { return "CopyGrad"; }
};
