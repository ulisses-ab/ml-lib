#pragma once

#include "../OpGradFn.h"

class MulGradFn : public OpGradFn {
public:
    MulGradFn(float val) : val_(val) {}


protected:
    void cpu_implementation(const Tensor grad_output) override;
    void cuda_implementation(const Tensor grad_output) override;

    std::string name() override { return "MulGrad"; }

private:
    float val_;
};