#pragma once

#include "../core/Tensor.h"
#include <vector>
#include <unordered_set>
#include "GradFn.h"
#include "../ops/fill/fill.h"

class AutogradEngine {
public:
    static void propagate(Tensor output) {
        fill_(output.grad(), 1.0f);

        std::vector<Tensor> topo;
        std::unordered_set<const void*> visited;

        build_topo(output, topo, visited);

        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            if (auto grad_fn = it->grad_fn().lock()) {
                Tensor g = it->grad();
                grad_fn->backward(g);
            }
        }
    }
private:
    static void build_topo(
        const Tensor current,
        std::vector<Tensor>& topo,
        std::unordered_set<const void*>& visited
    ) {
        const void* id = current.impl_ptr();

        if (visited.count(id)) return;
        visited.insert(id);

        auto grad_fn = current.grad_fn();
        if (auto p = grad_fn.lock()) {
            for (const Tensor& parent : p->parents()) {
                build_topo(parent, topo, visited);
            }
        }

        topo.push_back(current);
    }
};
