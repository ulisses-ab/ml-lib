#pragma once

#include "Module.h"
#include "../ops/gather/gather.h"
#include "../ops/rand_fill/rand_fill.h"
#include "../ops/add/add.h"

class Embedding : public Module {
public:
    Embedding(int vocab_size, int d_model, int max_seq_len = 2048, Device device = Device::CPU) 
        : Module(device), vocab_size_(vocab_size), d_model_(d_model), max_seq_len_(max_seq_len) {
        // Token embeddings: [vocab_size, d_model]
        Tensor token_emb(Shape({vocab_size, d_model}), device, true);
        rand_fill_(token_emb);
        register_parameter("token_embedding", token_emb);

        // Positional embeddings: [max_seq_len, d_model]
        Tensor pos_emb(Shape({max_seq_len, d_model}), device, true);
        rand_fill_(pos_emb);
        register_parameter("position_embedding", pos_emb);
    }

    Tensor forward(const std::vector<Tensor>& input) override {
        Tensor token_indices = input[0];
        Tensor token_emb = gather(token_indices, parameters_["token_embedding"]);
        

        int num_tokens = token_indices.numel();

        Tensor pos_emb = parameters_["position_embedding"];
        return add(token_emb, pos_emb.slice(0, 0, num_tokens).copy_to(device_).contiguous());
    }

private:
    int vocab_size_;
    int d_model_;
    int max_seq_len_;
};

