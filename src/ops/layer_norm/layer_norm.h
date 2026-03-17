#pragma once

#include "../../core/Tensor.h"

Tensor layer_norm(const Tensor x, const Tensor gamma, const Tensor beta, float eps = 1e-5f);

void layer_norm_(const Tensor x, const Tensor gamma, const Tensor beta, Tensor output, float eps = 1e-5f);


