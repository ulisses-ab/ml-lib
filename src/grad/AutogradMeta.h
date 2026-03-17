#pragma once

#include "../core/Storage.h"

class GradFn;

struct AutogradMeta {
    std::shared_ptr<Storage> data_;
    bool requires_grad_;
    std::shared_ptr<GradFn> grad_fn_;

    AutogradMeta(bool requires_grad, std::shared_ptr<GradFn> grad_fn, std::shared_ptr<Storage> data) :
        data_(data),
        requires_grad_(requires_grad),
        grad_fn_(grad_fn)
    {}
};