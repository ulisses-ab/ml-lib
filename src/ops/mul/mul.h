#pragma once

#include "../../core/Tensor.h"

Tensor mul(float val, const Tensor a);
Tensor mul(const Tensor& a, const Tensor& b);

void mul_(float val, const Tensor a, Tensor b);