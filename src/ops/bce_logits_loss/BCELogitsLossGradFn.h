#pragma once

#include "../OpGradFn.h"

class BCELogitsLossGradFn : public OpGradFn {
    using OpGradFn::OpGradFn;

    void cpu_implementation(const Tensor grad_output) override;
    void cuda_implementation(const Tensor grad_output) override;

    std::string name() override { return "BCELogitsLossGrad"; }
};
