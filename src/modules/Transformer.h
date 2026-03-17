#pragma once

#include <vector>
#include <memory>
#include <stdexcept>

#include "Module.h"
#include "EncoderBlock.h"
#include "DecoderBlock.h"

class Transformer : public Module {
public:
    Transformer(
        int d_model,
        int d_ffn,
        int num_layers,
        int num_heads = 8,
        Device device = Device::CPU
    )
        : Module(device),
          d_model_(d_model),
          d_ffn_(d_ffn),
          num_layers_(num_layers),
          num_heads_(num_heads)
    {
        if (num_layers_ <= 0) {
            throw std::invalid_argument("num_layers must be > 0");
        }

        for (int i = 0; i < num_layers_; ++i) {
            register_module(
                "encoder_" + std::to_string(i),
                std::make_shared<EncoderBlock>(
                    d_model_, d_ffn_, num_heads_, device
                )
            );
        }

        for (int i = 0; i < num_layers_; ++i) {
            register_module(
                "decoder_" + std::to_string(i),
                std::make_shared<DecoderBlock>(
                    d_model_, d_ffn_, num_heads_, device
                )
            );
        }
    }

    Tensor forward(const std::vector<Tensor>& input) override {
        if (input.size() != 2) {
            throw std::invalid_argument(
                "Transformer forward expects {src, tgt}"
            );
        }

        Tensor src = input[0];
        Tensor tgt = input[1];

        Tensor encoder_output = src;
        for (int i = 0; i < num_layers_; ++i) {
            encoder_output =
                submodules_["encoder_" + std::to_string(i)]
                    ->forward({encoder_output});
        }

        Tensor decoder_output = tgt;
        for (int i = 0; i < num_layers_; ++i) {
            decoder_output =
                submodules_["decoder_" + std::to_string(i)]
                    ->forward({decoder_output, encoder_output});
        }

        return decoder_output;
    }

private:
    int d_model_;
    int d_ffn_;
    int num_layers_;
    int num_heads_;
};
