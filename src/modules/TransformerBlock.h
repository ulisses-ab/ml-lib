#pragma once

#include "Module.h"
#include "MultiHeadAttention.h"
#include "MLP.h"
#include "LayerNorm.h"
#include "../ops/add/add.h"
#include "../ops/mask/mask.h"

class TransformerBlock : public Module {
public:
    TransformerBlock(
        int d_model,
        int num_heads,
        int d_ff,  // Feed-forward dimension
        Device device
    ) : Module(device), d_model_(d_model) {
        // Layer normalization before attention (pre-norm)
        auto ln1 = std::make_shared<LayerNorm>(d_model, 1e-5f, device);
        register_module("ln1", ln1);
        
        // Self-attention layer
        auto attn = std::make_shared<MultiHeadAttention>(d_model, num_heads, true, device);
        register_module("attention", attn);
        
        // Layer normalization before MLP (pre-norm)
        auto ln2 = std::make_shared<LayerNorm>(d_model, 1e-5f, device);
        register_module("ln2", ln2);
        
        // MLP (feed-forward network)
        // Typically: d_model -> d_ff -> d_model
        auto mlp = std::make_shared<MLP>(d_model, std::vector<int>{d_ff}, d_model, device);
        register_module("mlp", mlp);
    }

    Tensor forward(const std::vector<Tensor>& input) override {
        Tensor x = input[0];
        
        Tensor attn_output = submodules_["attention"]->forward({x, x, x});
        x = submodules_["ln1"]->forward(add(x, attn_output));
        
        attn_output = submodules_["mlp"]->forward({x});
        x = submodules_["ln2"]->forward(add(x, attn_output));
        
        return x;
    }
private:
    int d_model_;
};

