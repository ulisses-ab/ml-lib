#pragma once

#include "../../core/Tensor.h"

Tensor cross_entropy(const Tensor logits, const Tensor targets);