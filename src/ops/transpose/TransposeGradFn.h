#pragma once

#include "../OpGradFn.h"

class TransposeGradFn : public OpGradFn {
public:
    TransposeGradFn(int dim0, int dim1) : dim0_(dim0), dim1_(dim1) {}

    void cpu_implementation(const Tensor grad_output) override;
    void cuda_implementation(const Tensor grad_output) override;

    std::string name() override { return "TransposeGrad"; }

private:
    int dim0_;
    int dim1_;
};
