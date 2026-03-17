#pragma once

#include "../../core/Tensor.h"

Tensor mse(const Tensor a, const Tensor b);

void mse_(const Tensor a, const Tensor b, Tensor output);