#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>
#include <string>

#include "Module.h"
#include "../ops/matmul/matmul.h"
#include "../ops/softmax/softmax.h"
#include "../ops/rand_fill/rand_fill.h"
#include "../ops/attention/attention.h"
#include "../ops/concat/concat.h"

class MultiHeadAttention : public Module {
public:
    MultiHeadAttention(
        int d_model,
        int num_heads = 8,
        bool mask = false,
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

        for (int h = 0; h < num_heads_; ++h) {
            Tensor wqi(Shape({d_model, d_k_}), device, true);
            register_parameter("Wq_" + std::to_string(h), wqi);
            rand_fill_(wqi);

            Tensor wki(Shape({d_model, d_k_}), device, true);
            register_parameter("Wk_" + std::to_string(h), wki);
            rand_fill_(wki);

            Tensor wvi(Shape({d_model, d_v_}), device, true);
            register_parameter("Wv_" + std::to_string(h), wvi);
            rand_fill_(wvi);
        }

        Tensor wo(Shape({num_heads*d_v_, d_model}), device, true);
        register_parameter("Wo", wo);
        rand_fill_(wo);
    }

    Tensor forward(const std::vector<Tensor>& input) override {
        if (input.size() != 3) {
            throw std::invalid_argument("MultiHeadAttention expects Q, K, V");
        }

        const Tensor& Q = input[0];
        const Tensor& K = input[1];
        const Tensor& V = input[2];

        std::vector<Tensor> head_outputs;
        head_outputs.reserve(num_heads_);

        for (int h = 0; h < num_heads_; ++h) {
            Tensor Qh = matmul(Q, parameters_["Wq_" + std::to_string(h)]);
            Tensor Kh = matmul(K, parameters_["Wk_" + std::to_string(h)]);
            Tensor Vh = matmul(V, parameters_["Wv_" + std::to_string(h)]);

            head_outputs.push_back(attention(Qh, Kh, Vh, mask));
        }

        Tensor combined = concat(head_outputs);

        return matmul(combined, parameters_["Wo"]);
    }

private:
    int d_model_;
    int d_k_;
    int d_v_;
    int num_heads_;
};
