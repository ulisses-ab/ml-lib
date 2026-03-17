#pragma once

#include "../../core/Tensor.h"

Tensor add(const Tensor a, const Tensor b);

void add_(const Tensor a, const Tensor b, Tensor output);