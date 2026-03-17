#include "Layout.h"

#include <stdexcept>
#include <algorithm>

Layout::Layout(const Shape &shape) : 
    shape_(shape), 
    strides_(compute_default_strides(shape)), 
    base_offset_(0) 
{}

Layout::Layout(const Shape &shape, const std::vector<int> &strides, int base_offset) : 
    shape_(shape), 
    strides_(strides), 
    base_offset_(base_offset) 
{
    if (strides.size() != shape.num_dims()) {
        throw std::invalid_argument("Strides size must match number of dimensions in shape");
    }
}

const Shape &Layout::shape() const { return shape_; }

const std::vector<int> &Layout::strides() const {
    return strides_;
}

int Layout::size(int index) const { return shape_[index]; }

int Layout::stride(int index) const { return strides_[index]; }

int Layout::num_dims() const { return shape_.num_dims(); }

int Layout::numel() const { return shape_.numel(); }

int Layout::base_offset() const { return base_offset_; }

int Layout::offset(const std::vector<int> &indices) const {
  if (shape_.num_dims() == 0) {
    return 0;
  }

  int offset = base_offset_;
  for (int i = 0; i < indices.size(); ++i) {
    offset += indices[i] * strides_[i];
  }
  return offset;
}

std::vector<int> Layout::compute_default_strides(const Shape &shape) const {
    std::vector<int> strides(shape.num_dims());
    int stride = 1;
    for (int i = shape.num_dims() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape.size(i);
    }
    return strides;
}

bool Layout::is_contiguous() const {
  return strides_ == compute_default_strides(shape_);
}

Layout Layout::view(const Shape &new_shape) const {
    if (new_shape.numel() != shape_.numel()) {
        throw std::invalid_argument("New shape must have the same number of elements");
    }

    if (!is_contiguous()) {
        throw std::invalid_argument("Layout is not contiguous");
    }

    return Layout(new_shape);
}
 
Layout Layout::slice(int dim, int start, int end) const {
    Shape new_shape = shape_;
    new_shape[dim] = end - start;

    int new_base_offset = base_offset_ + start * strides_[dim];

    return Layout(new_shape, strides_, new_base_offset);
}

Layout Layout::transpose(int dim0, int dim1) const {
    Shape new_shape = shape_;
    std::swap(new_shape[dim0], new_shape[dim1]);

    std::vector<int> new_strides = strides_;
    std::swap(new_strides[dim0], new_strides[dim1]);

    return Layout(new_shape, new_strides, base_offset_);
}