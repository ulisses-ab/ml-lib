#pragma once

#include "Layout.h"
#include "Device.h"
#include "Storage.h"
#include "../grad/AutogradMeta.h"
#include <memory>

class Tensor;

class TensorImpl : public std::enable_shared_from_this<TensorImpl> {
public:
    TensorImpl(
        const Shape& shape, 
        Device device, 
        bool requires_grad, 
        std::shared_ptr<GradFn> grad_fn
    );

    // Acessors
    float* raw_data();
    float* data();
    float& at(const std::vector<int>& indices);

    template <typename... Args>
    float& at(Args... indices) {
        int offset = layout_.offset(indices...);
        return ((float*)storage_->data())[offset];
    }

    // Layout
    const Shape& shape();
    const Layout& layout();
    bool is_contiguous();

    // Device
    Device device();
    Tensor copy_to(Device device);

    // View
    Tensor cont();
    std::shared_ptr<TensorImpl> view(const Shape& new_shape);
    std::shared_ptr<TensorImpl> slice(int dim, int start, int end);
    std::shared_ptr<TensorImpl> transpose(int dim0, int dim1);

    // Grad
    bool requires_grad();
    std::shared_ptr<TensorImpl> grad();
    std::weak_ptr<GradFn> grad_fn();
    void clear_grad();
    std::shared_ptr<TensorImpl> detach();

    // Utils
    Tensor tensor();
private:
    TensorImpl(
        const Layout& layout, 
        std::shared_ptr<Storage> storage, 
        std::shared_ptr<AutogradMeta> grad_meta
    );

    const Layout layout_;
    const std::shared_ptr<Storage> storage_;
    const std::shared_ptr<AutogradMeta> autograd_meta_;
};