#include "Shape.h"

Shape::Shape(const std::vector<int> &dimensions) : dims_(dimensions) {}

int Shape::num_dims() const { return static_cast<int>(dims_.size()); }

int Shape::numel() const {
    int total = 1;
    for (int dim_size : dims_) {
        total *= dim_size;
    }
    return total;
}

int Shape::size(int index) const { return dims_.at(index); }

int& Shape::operator[](int index) { return dims_[index]; }

const int& Shape::operator[](int index) const { return dims_[index]; }

bool Shape::operator==(const Shape &other) const {
    return dims_ == other.dims_;
}

bool Shape::operator!=(const Shape &other) const {
    return dims_ != other.dims_;
}


