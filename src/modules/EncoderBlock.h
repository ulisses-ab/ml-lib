#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>
#include <string>
#include <memory>

#include "Module.h"
#include "../ops/matmul/matmul.h"
#include "../ops/softmax/softmax.h"
#include "../ops/rand_fill/rand_fill.h"
#include "../ops/attention/attention.h"
#include "../ops/concat/concat.h"
#include "MultiHeadAttention.h"
#include "MLP.h"
#include "LayerNorm.h"

class EncoderBlock : public Module {
public:
    EncoderBlock(
        int d_model,
        int d_ffn,
        int num_heads = 8,
        Device device = Device::CPU
    ) : 
        Module(device),
        d_model_(d_model),
        d_k_(d_model/num_heads),
        d_v_(d_model/num_heads),
        num_heads_(num_heads)
    {
        if (d_model_ % num_heads_ != 0) {
            throw std::invalid_argument("d_model must be divisible by num_heads");
        }

        register_module("Attention", std::make_shared<MultiHeadAttention>(d_model, num_heads, false, device));

        register_module("ln1", std::make_shared<LayerNorm>(d_model, 1e-5f, device));
        
        register_module("FeedForward", std::make_shared<MLP>(d_model, std::vector<int>{d_ffn}, d_model, device));

        register_module("ln2", std::make_shared<LayerNorm>(d_model, 1e-5f, device));
    }

    Tensor forward(const std::vector<Tensor>& input) override {
        Tensor x = input[0];
        
        Tensor attn_output = submodules_["Attention"]->forward({x, x, x});
        x = submodules_["ln1"]->forward(add(x, attn_output));
        
        attn_output = submodules_["FeedForward"]->forward({x});
        x = submodules_["ln2"]->forward(add(x, attn_output));
        
        return x;
    }

private:
    int d_model_;
    int d_k_;
    int d_v_;
    int num_heads_;
};
