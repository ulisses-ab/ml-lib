#pragma once

#include "../grad/GradFn.h"

class OpGradFn : public GradFn {
public:
    void backward(const Tensor grad_output) override {
        switch (grad_output.device()) {
            case Device::CPU:
                cpu_implementation(grad_output);
                break;
            case Device::CUDA:
                cuda_implementation(grad_output);
                break;
        }
    }
protected:
    virtual void cpu_implementation(const Tensor grad_output) = 0;
    virtual void cuda_implementation(const Tensor grad_output) = 0;
};