#pragma once

#include <memory>
#include "Device.h"
#include <string>
#include "TensorImpl.h"
#include <experimental/propagate_const>

class Tensor {
public:
    Tensor(
        const Shape& shape, 
        Device device = Device::CPU, 
        bool requires_grad = false, 
        std::shared_ptr<GradFn> grad_fn = nullptr
    );

    Tensor(std::shared_ptr<TensorImpl> impl);

    Tensor();

    // Acessors
    float* raw_data();
    float* data();
    const float* data() const;
    float& at(const std::vector<int>& indices);
    const float& at(const std::vector<int>& indices) const;

    template<typename... Args>
    float& operator()(Args... indices) {
        return impl_->at(indices...);
    }

    template<typename... Args>
    const float& operator()(Args... indices) const {
        return impl_->at(indices...);;
    }

    // Layout 
    const Shape& shape() const;
    const Layout& layout() const;
    int num_dims() const;
    int size(int dim) const;
    int numel() const;
    bool is_contiguous() const;

    // Device
    Device device() const;
    Tensor copy_to(Device device) const;

    // View
    Tensor contiguous() const;
    Tensor view(const Shape& new_shape) const;
    Tensor slice(int dim, int start, int end) const;
    Tensor narrow(int dim, int start, int length) const;
    Tensor transpose(int dim0, int dim1) const;
    
    // Grad
    bool requires_grad() const;
    Tensor grad();
    void clear_grad();
    Tensor detach();
    void backward();

    // Utils
    std::string to_string() const;
    void print() const;
    std::string shape_to_string() const;
    void print_shape() const;
    TensorImpl* impl_ptr();
    const TensorImpl* impl_ptr() const;
    bool empty() const;
    explicit operator bool() const;
private:
    friend class AutogradEngine;

    std::weak_ptr<GradFn> grad_fn() const;
    void print_recursive(std::stringstream& ss, int dim, std::vector<int>& indices) const;

    std::shared_ptr<TensorImpl> impl_;
};

