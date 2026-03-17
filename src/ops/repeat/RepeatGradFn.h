#pragma once

#include "../OpGradFn.h"

class RepeatGradFn : public OpGradFn {
    void cpu_implementation(const Tensor grad_output) override;
    void cuda_implementation(const Tensor grad_output) override;
    
    std::string name() override { return "RepeatGrad"; }
};