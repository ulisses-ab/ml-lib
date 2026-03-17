#pragma once

#include "Shape.h"
#include <vector>

class Layout {
public:
    Layout(const Shape& shape);
    Layout(const Shape& shape, const std::vector<int>& strides, int base_offset = 0);

    const Shape& shape() const;
    const std::vector<int>& strides() const;

    int size(int index) const;
    int stride(int index) const;

    int num_dims() const;
    int numel() const;

    int base_offset() const;
    int offset(const std::vector<int>& indices) const;

    template <typename... Args>
    int offset(Args... indices) const {
        if (shape_.num_dims() == 0) {
            return 0;
        }

        int offset = base_offset_;
        int i = 0;

        ((offset += static_cast<int>(indices) * strides_[i++]), ...);

        return offset;        
    }

    bool is_contiguous() const;

    Layout view(const Shape& new_shape) const;
    Layout slice(int dim, int start, int end) const;
    Layout transpose(int dim0, int dim1) const;
private:
    std::vector<int> compute_default_strides(const Shape& shape) const;

    Shape shape_;
    std::vector<int> strides_;
    int base_offset_;
};