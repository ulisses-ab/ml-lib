#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include "../core/Tensor.h"

class Module {
public:
    Module(Device device) : device_(device) {}
    virtual ~Module() = default;

    Tensor forward(Tensor input) {
        return forward(std::vector<Tensor>{input});
    }

    virtual Tensor forward(const std::vector<Tensor>& input) = 0;

    std::map<std::string, Tensor> param_map() {
        std::map<std::string, Tensor> all_params = parameters_;
        for (auto const& [name, module] : submodules_) {
            auto sub_params = module->param_map();
            for (auto const& [p_name, p_tensor] : sub_params) {
                all_params[name + "." + p_name] = p_tensor;
            }
        }
        return all_params;
    }

    std::vector<Tensor> parameters() {
        std::vector<Tensor> output;

        auto params = param_map();

        for (auto const& [_, p] : params) {
            output.push_back(p);
        }

        return output;
    }

protected:
    void register_parameter(const std::string& name, Tensor param) {
        parameters_[name] = param;
    }

    void register_module(const std::string& name, std::shared_ptr<Module> module) {
        submodules_[name] = module;
    }

    Device device_;
    std::map<std::string, Tensor> parameters_;
    std::map<std::string, std::shared_ptr<Module>> submodules_;
};
