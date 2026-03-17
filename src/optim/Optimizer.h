#pragma once

#include "../core/Tensor.h"

class Optimizer {
protected:
    std::vector<Tensor> parameters;

public:
    Optimizer(const std::vector<Tensor>& params) : 
        parameters(params) 
    {}

    void zero_grad() {
        for (auto p : parameters) {
            p.clear_grad();
        }
    }

    virtual void step(float lr) = 0;
};