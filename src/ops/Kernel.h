#pragma once

#include "../core/Tensor.h"
#include <vector>
#include <stdexcept>
#include <iostream>

class Kernel {
public:
    Kernel() : requires_contiguous_(true) {}
    Kernel(bool requires_contiguous) : requires_contiguous_(requires_contiguous) {}

    void compute(const std::vector<Tensor>& inputs, Tensor output) {
        if (requires_contiguous_) {
            certify_contiguous(inputs, output);
        }

        validate(inputs);

        if (output.shape() != get_output_shape(inputs)) {
            throw std::invalid_argument("Kernel: output doesnt fit the requires shape");
        }
        
        Device device = get_device(inputs, output);

        switch (device) {
            case Device::CPU:
                cpu_implementation(inputs, output);
                break;
            case Device::CUDA:
                cuda_implementation(inputs, output);
                break;
        }
    }

    virtual Shape get_output_shape(const std::vector<Tensor>& inputs) = 0;

protected:
    virtual void validate(const std::vector<Tensor>& inputs) = 0;
    virtual void cpu_implementation(const std::vector<Tensor>& inputs, Tensor output) = 0;
    virtual void cuda_implementation(const std::vector<Tensor>& inputs, Tensor output) = 0;

private:
    bool requires_contiguous_ = true;

    void certify_contiguous(const std::vector<Tensor>& inputs, Tensor output) {
        for (const auto& input : inputs) {
            if (!input.is_contiguous()) {
                throw std::invalid_argument("kernel error: all inputs must be contiguous " + input.to_string());
            }
        }

        if (!output.is_contiguous()) {
            throw std::invalid_argument("kernel error: output must be contiguous");
        }
    }

    Device get_device(const std::vector<Tensor>& inputs, Tensor output) {
        Device device = inputs.at(0).device();

        for (const auto& input : inputs) {
            if (input.device() != device) {
                throw std::invalid_argument("kernel error: all inputs must be on the same device");
            }
        }

        if (output.device() != device) {
            throw std::invalid_argument("kernel error: output must be on the same device as inputs");
        }

        return device;
    }
};