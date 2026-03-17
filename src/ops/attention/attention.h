#pragma once

#include <cmath>
#include <stdexcept>

#include "../../core/Tensor.h"
#include "../matmul/matmul.h"
#include "../softmax/softmax.h"
#include "../mul/mul.h"
#include <iostream>
#include "../transpose/transpose.h"
#include "../mask/mask.h"

inline Tensor attention(
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    bool apply_mask = false
) {
    if (Q.shape()[1] != K.shape()[1]) {
        throw std::invalid_argument(
            "sdp_attention: Q and K must have the same last dimension"
        );
    }

    int d_k = Q.shape()[1];

    Tensor Kt = transpose(K, 0, 1).contiguous();

    Tensor scores = matmul(Q, Kt);

    float scale = 1.0f / std::sqrt(static_cast<float>(d_k));
    scores = mul(scale, scores);

    if (apply_mask) {
        scores = mask(scores);
    }

    Tensor attn = softmax(scores);

    return matmul(attn, V);
}