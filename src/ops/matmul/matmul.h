#pragma once

#include "../../core/Tensor.h"

Tensor matmul(const Tensor a, const Tensor b);

void matmul_(const Tensor a, const Tensor b, Tensor output);