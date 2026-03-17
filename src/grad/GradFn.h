#pragma once

#include <memory>
#include <vector>
#include <string>
#include "../core/Tensor.h"

class GradFn {
public:
    void add_parents(const std::vector<Tensor>& parents) {
        for (auto p : parents) {
            parents_.push_back(p);
        }
    }

    const std::vector<Tensor>& parents() const {
        return parents_;
    };

    virtual void backward(const Tensor grad_output) = 0;
    virtual std::string name() = 0;
protected:
    std::vector<Tensor> parents_;
};