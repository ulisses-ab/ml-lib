#pragma once

#include "Optimizer.h"
#include "../ops/mul/mul.h"
#include "../ops/add/add.h"

class SGD : public Optimizer {
public:
    using Optimizer::Optimizer;

    void step(float lr) override {
        for (auto p : parameters) {
            add_(p, mul(-lr, p.grad()), p);
        }
    }
};