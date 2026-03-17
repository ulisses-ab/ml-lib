#pragma once

#include "../OpGradFn.h"

class LayerNormGradFn : public OpGradFn {
public:
    LayerNormGradFn(float eps) : eps_(eps) {}

    void cpu_implementation(const Tensor grad_output) override;
    void cuda_implementation(const Tensor grad_output) override;

    std::string name() override { return "LayerNormGrad"; }

private:
    float eps_;
};


