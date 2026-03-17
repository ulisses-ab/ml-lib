#include "copy.h"
#include "CopyGradFn.h"
#include <cuda_runtime.h>
#include <cstring>
#include <vector>
#include <memory>

Tensor copy(const Tensor a, Device device) {
    bool requires_grad = a.requires_grad();
    std::shared_ptr<GradFn> grad_fn = nullptr;
    if(requires_grad) {
        grad_fn = std::make_shared<CopyGradFn>();
        grad_fn->add_parents(std::vector<Tensor>{a});
    }

    Tensor output(a.shape(), device, requires_grad, grad_fn);

    Tensor src = a.contiguous();

    void* src_ptr = src.data();
    void* dst_ptr = output.data();
    size_t size = src.numel() * sizeof(float);

    Device src_dev = src.device();
    Device dst_dev = device;

    if (src_dev == Device::CPU && dst_dev == Device::CPU) {
        std::memcpy(dst_ptr, src_ptr, size);
    } else if (src_dev == Device::CPU && dst_dev == Device::CUDA) {
        cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyHostToDevice);
    } else if (src_dev == Device::CUDA && dst_dev == Device::CPU) {
        cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDeviceToHost);
    } else if (src_dev == Device::CUDA && dst_dev == Device::CUDA) {
        cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDeviceToDevice);
    }

    return output;
}
