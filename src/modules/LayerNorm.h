#pragma once

#include "Module.h"
#include "../ops/layer_norm/layer_norm.h"
#include "../ops/rand_fill/rand_fill.h"
#include "../ops/fill/fill.h"

class LayerNorm : public Module {
public:
    LayerNorm(int d_model, float eps = 1e-5f, Device device = Device::CPU) 
        : Module(device), d_model_(d_model), eps_(eps) {
        // Gamma (scale) parameter - initialized to ones
        Tensor gamma(Shape({d_model}), device, true);
        fill_(gamma, 1.0f);
        register_parameter("gamma", gamma);

        // Beta (shift) parameter - initialized to zeros
        Tensor beta(Shape({d_model}), device, true);
        fill_(beta, 0.0f);
        register_parameter("beta", beta);
    }

    Tensor forward(const std::vector<Tensor>& input) override {
        Tensor x = input[0];
        return layer_norm(x, parameters_["gamma"], parameters_["beta"], eps_);
    }

private:
    int d_model_;
    float eps_;
};

