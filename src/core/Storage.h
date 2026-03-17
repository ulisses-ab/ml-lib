#pragma once

#include "Device.h"
#include <cstddef>
#include <memory>

class Storage {
public:
    Storage(int size, Device device);
    ~Storage();

    void* data() const;
    int size() const;
    Device device() const;

    std::shared_ptr<Storage> copy_to(Device device) const;
private:
    void allocate();
    void deallocate();

    void* data_;
    int size_;
    Device device_;
};