#pragma once

#include "Module.h"
#include "../ops/gather/gather.h"
#include "../ops/rand_fill/rand_fill.h"
#include "../ops/add/add.h"
#include "../ops/concat/concat.h" 

class DCNEmbedding : public Module {
public:
    DCNEmbedding(std::vector<int> category_sizes, int d_model, Device device = Device::CPU) 
        : Module(device), category_sizes_(category_sizes), d_model_(d_model) {

        int i = 0;
        for (int size : category_sizes) {
            Tensor weight(Shape({size, d_model}), device, true);
            rand_fill_(weight);
            register_parameter("embedding_" + std::to_string(i), weight);
            i++;
        }
    }

    Tensor forward(const std::vector<Tensor>& input) override {
        Tensor x = input[0];
        Tensor dense = input[1];
        Tensor out(Shape({1, 0}));

        out = concat({out, dense});

        for (int i = 0; i < x.numel(); i++) {
            Tensor temp(Shape({1}));
            temp.at({0}) = x(0, i);
            temp = temp.copy_to(device_);

            Tensor emb = gather(temp, parameters_["embedding_" + std::to_string(i)]);
            out = concat({out, emb});
        }

        return out.view(Shape({out.size(1)}));
    }

private:
    std::vector<int> category_sizes_;
    int d_model_;
};

