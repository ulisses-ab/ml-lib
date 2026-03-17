#include "rand_fill.h"
#include "../Kernel.h"
#include <stdexcept>
#include <random>

void launch_rand_fill_cuda(float* data, int n, unsigned long long seed);

class RandFillKernel : public Kernel {
public:
    Shape get_output_shape(const std::vector<Tensor>& inputs) override {
        return inputs.at(0).shape();
    }

protected:
    void validate(const std::vector<Tensor>& inputs) override {
        if (inputs.size() != 1) {
            throw std::invalid_argument("randfill: must have one input");
        }
    }

    void cpu_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        float* A = output.data();
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-0.1f, 0.1f);

        for (int i = 0; i < output.numel(); i++) {
            A[i] = dis(gen);
        }
    }

    void cuda_implementation(const std::vector<Tensor>& inputs, Tensor output) override {
        std::random_device rd;
        unsigned long long seed = rd();
        launch_rand_fill_cuda(output.data(), output.numel(), seed);
    }
};

void rand_fill_(Tensor a) {
    RandFillKernel kernel;
    // We only pass the tensor 'a' as both the shape reference and output
    kernel.compute({a}, a);
}