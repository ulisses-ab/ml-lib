#pragma once

#include <vector>
#include "Module.h"
#include "Linear.h"
#include "../ops/relu/relu.h" 

class MLP : public Module {
public:
    MLP(int input_size, const std::vector<int>& hidden_sizes, int output_size, Device device) : Module(device) {
        int current_in = input_size;

        for (size_t i = 0; i < hidden_sizes.size(); ++i) {
            std::string name = "hidden_" + std::to_string(i);
            auto layer = std::make_shared<Linear>(current_in, hidden_sizes[i], device);
            
            register_module(name, layer);
            layers_.push_back(layer);
            
            current_in = hidden_sizes[i];
        }

        auto out_layer = std::make_shared<Linear>(current_in, output_size, device);
        register_module("output", out_layer);
        layers_.push_back(out_layer);
    }

    Tensor forward(const std::vector<Tensor>& input) override {
        Tensor x = input[0];

        for (size_t i = 0; i < layers_.size(); ++i) {
            x = layers_[i]->forward(std::vector<Tensor>{x});
            
            if (i < layers_.size() - 1) {
                x = relu(x);
            }
        }

        return x;
    }

private:
    std::vector<std::shared_ptr<Module>> layers_;
};