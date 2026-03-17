#pragma once

#include "Module.h"
#include "../ops/matmul/matmul.h"
#include "../ops/add/add.h"
#include "../ops/mul/mul.h"
#include "../ops/repeat/repeat.h"
#include "../ops/rand_fill/rand_fill.h"

class CrossLayer : public Module {
public:
    CrossLayer(int dim, Device device = Device::CPU) : Module(device), dim_(dim) {
        Tensor w(Shape({dim, dim}), device, true);
        Tensor b(Shape({dim}), device, true);

        rand_fill_(w);
        rand_fill_(b);

        register_parameter("w", w);
        register_parameter("b", b);
    }

    Tensor forward(const std::vector<Tensor>& input) override {
        Tensor x = input[0];
        Tensor x0 = (input.size() > 1) ? input[1] : input[0];
        
        Tensor wx = matmul(x.view(Shape({1, x.numel()})), parameters_["w"]);

        Tensor bias = parameters_["b"];
        Tensor projection = add(wx.view(Shape({wx.numel()})), bias);

        Tensor gated = mul(x0, projection);

        Tensor out = add(x, gated);
        return out;
    }

private:
    int dim_;
};