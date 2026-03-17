#pragma once

#include "../../core/Tensor.h"

Tensor softmax(const Tensor a);

void softmax_(const Tensor a, Tensor b);