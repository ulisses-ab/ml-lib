#pragma once

#include "Optimizer.h"
#include "../ops/mul/mul.h"
#include "../ops/add/add.h"
#include "../ops/fill/fill.h"
#include "../ops/sqrt/sqrt.h"  
#include "../ops/div/div.h"

#include <unordered_map>
#include <cmath>

class Adam : public Optimizer {
public:
    Adam(
        const std::vector<Tensor>& params,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float eps   = 1e-8f
    )
        : Optimizer(params),
          beta1(beta1),
          beta2(beta2),
          eps(eps),
          t(0)
    {
        // Initialize m and v for each parameter
        for (auto& p : parameters) {
            Tensor a(p.shape(), p.device());
            Tensor b(p.shape(), p.device());
            fill_(a, 0.0f);
            fill_(b, 0.0f);
            m[p.impl_ptr()] = a;
            v[p.impl_ptr()] = b;
        }
    }

    void step(float lr) override {
        t += 1;

        for (auto& p : parameters) {
            auto& mt = m[p.impl_ptr()];
            auto& vt = v[p.impl_ptr()];
            Tensor g = p.grad();

            // m = beta1 * m + (1 - beta1) * g
            add_(
                mul(beta1, mt),
                mul(1.0f - beta1, g),
                mt
            );
            

            // v = beta2 * v + (1 - beta2) * g^2
            add_(
                mul(beta2, vt),
                mul(1.0f - beta2, mul(g, g)),
                vt
            );

            
            // Bias correction
            float bias_correction1 = 1.0f - std::pow(beta1, t);
            float bias_correction2 = 1.0f - std::pow(beta2, t);

            Tensor m_hat = mul(1.0f/bias_correction1, mt);
            Tensor v_hat = mul(1.0f/bias_correction2, vt);

            // p -= lr * m_hat / (sqrt(v_hat) + eps)
            Tensor vhc = sqrt(v_hat);

            
            Tensor epst(vhc.shape(), vhc.device());
            fill_(epst, eps);

            Tensor denom = add(vhc, epst);
            Tensor step = div(m_hat, denom);

            add_(p, mul(-lr, step), p);
        }
    }

private:
    float beta1, beta2, eps;
    int t;

    // Parameter state (keyed by Tensor id)
    std::unordered_map<const void *, Tensor> m;
    std::unordered_map<const void *, Tensor> v;
};
