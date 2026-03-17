#pragma once

#include "../../core/Tensor.h"

Tensor relu(const Tensor a);

void relu_(const Tensor a, Tensor output);