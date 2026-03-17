#include "TensorImpl.h"
#include "Tensor.h"
#include "../ops/fill/fill.h"
#include "../ops/contiguous/contiguous.h"
#include "../ops/copy/copy.h"
#include "stdexcept"

TensorImpl::TensorImpl(const Shape &shape, Device device, bool requires_grad, std::shared_ptr<GradFn> grad_fn) :
    layout_(shape),
    storage_(std::make_shared<Storage>(shape.numel()*sizeof(float), device)),
    autograd_meta_(std::make_shared<AutogradMeta>(requires_grad, grad_fn, nullptr))
{}

TensorImpl::TensorImpl(const Layout& layout, std::shared_ptr<Storage> storage, std::shared_ptr<AutogradMeta> autograd_meta) :
    layout_(layout),
    storage_(storage),
    autograd_meta_(autograd_meta)
{}

float* TensorImpl::raw_data() {
    return ((float*)storage_->data()) + layout_.base_offset(); 
}

float* TensorImpl::data() { 
    if (!is_contiguous()) {
        throw std::runtime_error("Cannot get raw pointer to non-contiguous Tensor. Call .contiguous() first.");
    }
    return ((float*)storage_->data()) + layout_.base_offset(); 
}

float& TensorImpl::at(const std::vector<int> &indices) {
    int offset = layout_.offset(indices);
    return ((float*)storage_->data())[offset];
}

const Shape& TensorImpl::shape() {
    return layout_.shape();
}

const Layout &TensorImpl::layout() {
    return layout_;
}

bool TensorImpl::is_contiguous() { return layout_.is_contiguous(); }

Device TensorImpl::device() { 
    return storage_->device(); 
}

Tensor TensorImpl::copy_to(Device device) {
    return copy(tensor(), device);
}

Tensor TensorImpl::cont() {
    if (is_contiguous()) return tensor();

    return contiguous(tensor());
}

std::shared_ptr<TensorImpl> TensorImpl::view(const Shape &new_shape) {
    Layout new_layout = layout_.view(new_shape);
    return std::shared_ptr<TensorImpl>(
        new TensorImpl(new_layout, storage_, autograd_meta_)
    );
}

std::shared_ptr<TensorImpl> TensorImpl::slice(int dim, int start, int end) {
    Layout new_layout = layout_.slice(dim, start, end);
    return std::shared_ptr<TensorImpl>(new TensorImpl(new_layout, storage_, autograd_meta_));
}

std::shared_ptr<TensorImpl> TensorImpl::transpose(int dim0, int dim1) {
    Layout new_layout = layout_.transpose(dim0, dim1);
    return std::shared_ptr<TensorImpl>(new TensorImpl(new_layout, storage_, autograd_meta_));
}

bool TensorImpl::requires_grad() {
    return autograd_meta_->requires_grad_;    
}

std::shared_ptr<TensorImpl> TensorImpl::grad() {
    if(!requires_grad()) {
        throw std::runtime_error("called grad() on a tensor that doesnt require grad");
    }

    auto& data = autograd_meta_->data_;

    if(!data) {
        data = std::make_shared<Storage>(shape().numel()*sizeof(float), device());
        clear_grad();
    }

    return std::shared_ptr<TensorImpl>(new TensorImpl(layout_, data, 
        std::make_shared<AutogradMeta>(false, nullptr, nullptr)
    ));
}

std::weak_ptr<GradFn> TensorImpl::grad_fn() { 
    return autograd_meta_->grad_fn_;
}

void TensorImpl::clear_grad() {
    Tensor t = tensor();
    Tensor grad = t.grad();
    fill_(grad, 0.0f);
}

std::shared_ptr<TensorImpl> TensorImpl::detach() {
    return std::shared_ptr<TensorImpl>(new TensorImpl(layout_, storage_, 
        std::make_shared<AutogradMeta>(false, nullptr, nullptr)
    ));
}

Tensor TensorImpl::tensor() { return Tensor(shared_from_this()); }
