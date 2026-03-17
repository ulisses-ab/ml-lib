#include "Storage.h"

#include <cuda_runtime.h>
#include <cstring>
#include <stdlib.h>
#include <iostream>

Storage::Storage(int size, Device device) :
    size_(size), 
    device_(device),
    data_(nullptr)
{
    allocate();
}

Storage::~Storage() {
    deallocate();
}

void* Storage::data() const { return data_; }

int Storage::size() const { return size_; }

Device Storage::device() const { return device_; }

std::shared_ptr<Storage> Storage::copy_to(Device device) const {
    auto new_storage = std::make_shared<Storage>(size_, device);

    if (device_ == Device::CPU && device == Device::CPU) {
        std::memcpy(new_storage->data(), data_, size_);
    } else if (device_ == Device::CPU && device == Device::CUDA) {
        cudaMemcpy(new_storage->data(), data_, size_, cudaMemcpyHostToDevice);
    } else if (device_ == Device::CUDA && device == Device::CPU) {
        cudaMemcpy(new_storage->data(), data_, size_, cudaMemcpyDeviceToHost);
    } else if (device_ == Device::CUDA && device == Device::CUDA) {
        cudaMemcpy(new_storage->data(), data_, size_, cudaMemcpyDeviceToDevice);
    }

    return new_storage;
}

void Storage::allocate() {
    if (device_ == Device::CPU) {
        data_ = std::malloc(size_);
    } else if (device_ == Device::CUDA) {
        cudaMalloc(&data_, size_);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error (allocation): " << cudaGetErrorString(err) << std::endl;
        }
    }
}

void Storage::deallocate() {
    if (device_ == Device::CPU) {
        std::free(data_);
    } else if (device_ == Device::CUDA) {
        cudaFree(data_);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error (deallocation): " << cudaGetErrorString(err) << std::endl;
        }
    }
}