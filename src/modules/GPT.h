#pragma once

#include <vector>
#include <memory>
#include "Module.h"
#include "Embedding.h"
#include "TransformerBlock.h"
#include "Linear.h"
#include "../ops/softmax/softmax.h"

class GPT : public Module {
public:
    GPT(
        int vocab_size,
        int d_model,
        int num_heads,
        int num_layers,
        int d_ff,
        int max_seq_len = 2048,
        Device device = Device::CPU
    ) : Module(device),
        vocab_size_(vocab_size),
        d_model_(d_model),
        num_heads_(num_heads),
        num_layers_(num_layers),
        max_seq_len_(max_seq_len)
    {
        // Token and positional embeddings
        auto embedding = std::make_shared<Embedding>(vocab_size, d_model, max_seq_len, device);
        register_module("embedding", embedding);
        
        // Transformer blocks
        for (int i = 0; i < num_layers; ++i) {
            auto block = std::make_shared<TransformerBlock>(d_model, num_heads, d_ff, device);
            register_module("transformer_" + std::to_string(i), block);
            transformer_blocks_.push_back(block);
        }
        
        // Output projection to vocabulary
        auto lm_head = std::make_shared<Linear>(d_model, vocab_size, device);
        register_module("lm_head", lm_head);
    }

    Tensor forward(const std::vector<Tensor>& input) override {
        return softmax(forward_logits(input));
    }

    // Forward pass that returns logits (before softmax) - useful for training
    Tensor forward_logits(const std::vector<Tensor>& input) {
        Tensor x = submodules_["embedding"]->forward({input[0]});
        
        for (auto& block : transformer_blocks_) {
            x = block->forward({x});
        }

        x = submodules_["lm_head"]->forward({x});

        return x;
    }

private:
    int vocab_size_;
    int d_model_;
    int num_heads_;
    int num_layers_;
    int max_seq_len_;
    std::vector<std::shared_ptr<TransformerBlock>> transformer_blocks_;
};

