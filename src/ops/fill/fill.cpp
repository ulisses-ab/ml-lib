#include "fill.h"
#include "../Kernel.h"
#include <stdexcept>

void launch_fill_cuda(float* data, float val, int n);

void fill_(Tensor a, float val) {
    if (a.device() == Device::CPU) {
        float* A = a.raw_data();

        for (int i = 0; i < a.numel(); i++) {
            A[i] = val;
        }
    }
    else if (a.device() == Device::CUDA) {
        launch_fill_cuda(a.raw_data(), val, a.numel());
    }
}