#include "Tensor.h"

#include <sstream>
#include <iomanip>
#include <iostream>
#include "../grad/AutogradEngine.h"

Tensor::Tensor(const Shape &shape, Device device, bool requires_grad, std::shared_ptr<GradFn> grad_fn) :
    impl_(std::make_shared<TensorImpl>(shape, device, requires_grad, grad_fn))
{}

Tensor::Tensor(std::shared_ptr<TensorImpl> impl) :
    impl_(impl)
{}

Tensor::Tensor() : impl_(nullptr) {}

float* Tensor::raw_data() { return impl_->raw_data(); }

float* Tensor::data() { return impl_->data(); }

const float* Tensor::data() const { return impl_->data(); }

float& Tensor::at(const std::vector<int> &indices) {
  return impl_->at(indices);
}

const float& Tensor::at(const std::vector<int> &indices) const {
  return impl_->at(indices);
}

const Shape& Tensor::shape() const { return impl_->shape(); }

const Layout &Tensor::layout() const { return impl_->layout(); }

int Tensor::num_dims() const { return impl_->shape().num_dims(); }

int Tensor::size(int dim) const { return impl_->shape().size(dim); }

int Tensor::numel() const { return impl_->shape().numel(); }

bool Tensor::is_contiguous() const { return impl_->is_contiguous(); }

Device Tensor::device() const {
    return impl_->device();
}

Tensor Tensor::copy_to(Device device) const { 
    return impl_->copy_to(device);    
}

Tensor Tensor::contiguous() const { return impl_->cont(); }

Tensor Tensor::view(const Shape &new_shape) const { return Tensor(impl_->view(new_shape)).contiguous(); }

Tensor Tensor::slice(int dim, int start, int end) const { 
    return Tensor(impl_->slice(dim, start, end)).contiguous();    
}

Tensor Tensor::transpose(int dim0, int dim1) const { return Tensor(impl_->transpose(dim0, dim1)).contiguous(); }

bool Tensor::requires_grad() const { return impl_->requires_grad(); }

Tensor Tensor::grad() { return Tensor(impl_->grad()); }

void Tensor::clear_grad() { impl_->clear_grad(); }

Tensor Tensor::detach() { return Tensor(impl_->detach()); }

void Tensor::backward() {
    AutogradEngine::propagate(*this);
}

std::string Tensor::to_string() const {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(4);
    
    std::vector<int> indices(num_dims(), 0);
    
    ss << "Tensor(";
    print_recursive(ss, 0, indices);
    ss << ", device=" << (device() == Device::CPU ? "CPU" : "CUDA") << ")";
    
    return ss.str();
}

std::weak_ptr<GradFn> Tensor::grad_fn() const { return impl_->grad_fn(); }

void Tensor::print_recursive(
    std::stringstream &ss, 
    int dim,
    std::vector<int> &indices
) const {
    if (device() != Device::CPU) {
        return copy_to(Device::CPU).print_recursive(ss, dim, indices);
    }

    int dim_size = impl_->shape()[dim];

    if (dim == num_dims() - 1) {
        // Base case: print the actual row of numbers
        ss << "[";
        for (int i = 0; i < dim_size; ++i) {
            indices[dim] = i;
            ss << at(indices);
            if (i < dim_size - 1) {
                ss << ", ";
            }
        }
        ss << "]";
    } else {
        // Recursive case: print nested brackets
        ss << "[";
        for (int i = 0; i < dim_size; ++i) {
            indices[dim] = i;
            print_recursive(ss, dim + 1, indices);
            if (i < dim_size - 1) {
                ss << ",\n" << std::string(dim + 8, ' '); // Alignment padding
            }
        }
        ss << "]";
    }
}

void Tensor::print() const {
    std::cout << to_string() << std::endl;
}

std::string Tensor::shape_to_string() const {
    std::string out = "[";

    for (int i = 0; i < num_dims(); i++) {
        out += std::to_string(size(i));
        if (i < num_dims()-1) {
            out += ", ";
        }
    }

    out += "]";

    return out;
}

void Tensor::print_shape() const {
    std::cout << shape_to_string() << std::endl;
}

TensorImpl* Tensor::impl_ptr() { return impl_.get(); }

const TensorImpl* Tensor::impl_ptr() const { return impl_.get(); }

bool Tensor::empty() const { return impl_.get() == nullptr; }

Tensor::operator bool() const { return impl_.get() != nullptr; }
