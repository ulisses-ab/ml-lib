#pragma once

#include <cstddef>
#include <initializer_list>
#include <vector>

class Shape {
public:
    Shape(const std::vector<int>& dimensions);

    int num_dims() const;
    int numel() const;

    int size(int index) const;

    int& operator[](int index);
    const int& operator[](int index) const;

    bool operator==(const Shape& other) const;
    bool operator!=(const Shape& other) const;
private:
    std::vector<int> dims_;
};