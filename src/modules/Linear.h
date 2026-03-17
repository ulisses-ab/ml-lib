#pragma once

#include "Module.h"
#include "../ops/matmul/matmul.h"
#include "../ops/add/add.h"
#include "../ops/rand_fill/rand_fill.h"
#include "../ops/repeat/repeat.h"

class Linear : public Module {
public:
    Linear(int in_features, int out_features, Device device) : Module(device) {
        Tensor weights(Shape({in_features, out_features}), device, true);
        Tensor bias(Shape({out_features}), device, true);

        rand_fill_(weights);
        rand_fill_(bias);

        register_parameter("weight", weights);
        register_parameter("bias", bias);
    }

    Tensor forward(const std::vector<Tensor>& input) override {
        Tensor x = input[0];
        Tensor bias = repeat(x.shape()[0], parameters_["bias"]);
        Tensor output = matmul(x, parameters_["weight"]);
        output = add(output, bias);
        return output;
    }
};

