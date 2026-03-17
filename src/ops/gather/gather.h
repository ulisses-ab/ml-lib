#pragma once

#include "../../core/Tensor.h"

Tensor gather(const Tensor indices, const Tensor embedding);

void gather_(const Tensor indices, const Tensor embedding, Tensor output);

