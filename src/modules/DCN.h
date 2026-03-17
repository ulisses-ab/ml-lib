#pragma once

#include "Module.h"
#include "CrossLayer.h"
#include "MLP.h"
#include "../ops/add/add.h"
#include <vector>
#include <memory>

class DCN : public Module {
public:
    DCN(int input_dim, int num_cross_layers, const std::vector<int>& tower_hidden, int tower_output_dim, Device device = Device::CPU)
        : Module(device), input_dim_(input_dim) {
        // create cross layers
        for (int i = 0; i < num_cross_layers; ++i) {
            auto layer = std::make_shared<CrossLayer>(input_dim, device);
            register_module("cross_" + std::to_string(i), layer);
            cross_layers_.push_back(layer);
        }

        // create tower MLP
        tower_ = std::make_shared<MLP>(input_dim, tower_hidden, tower_output_dim, device);
        register_module("tower", tower_);
    }

    Tensor forward(const std::vector<Tensor>& input) override {
        Tensor x = input[0];
        Tensor x0 = x;

        // cross network: feed x through cross layers, passing original x0
        for (size_t i = 0; i < cross_layers_.size(); ++i) {
            x = cross_layers_[i]->forward(std::vector<Tensor>{x, x0});
        }

        // pass result to tower
        Tensor tower_out = tower_->forward(std::vector<Tensor>{x.view(Shape({1, input_dim_}))});

        return tower_out;
    }

private:
    int input_dim_;
    std::vector<std::shared_ptr<CrossLayer>> cross_layers_;
    std::shared_ptr<MLP> tower_;
};
