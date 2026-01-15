#include "tensor.h"
#include "utils.h"
#include <vector>
#include <cstring>
#include <cassert>
#include <cuda_runtime.h>

size_t prod_shape(const std::vector<size_t>& s) {
    size_t total = 1;
    for (size_t x : s) total *= x;
    return total;
}

// ------------------ Tensor class ------------------

Tensor::Tensor() : shape_({0}), size_(0), device_(Device::CPU), host_ptr_(nullptr), dev_ptr_(nullptr) {}

Tensor::Tensor(const std::vector<size_t>& shape, Device device)
    : shape_(shape), size_(prod_shape(shape)), device_(device), host_ptr_(nullptr), dev_ptr_(nullptr) {
    allocate();
}

// Copy constructor
Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), size_(other.size_), device_(other.device_),
      host_ptr_(nullptr), dev_ptr_(nullptr) {
    if (size_ > 0) {
        if (device_ == Device::CPU) {
            allocate_host();
            std::memcpy(host_ptr_, other.host_ptr_, sizeof(float) * size_);
        } else {
            allocate_device();
            CUDA_CHECK(cudaMemcpy(dev_ptr_, other.dev_ptr_, sizeof(float) * size_, cudaMemcpyDeviceToDevice));
        }
    }
}

// Assignment operator
Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) return *this;
    free_storage();
    shape_ = other.shape_;
    size_ = other.size_;
    device_ = other.device_;
    if (size_ > 0) {
        if (device_ == Device::CPU) {
            allocate_host();
            std::memcpy(host_ptr_, other.host_ptr_, sizeof(float) * size_);
        } else {
            allocate_device();
            CUDA_CHECK(cudaMemcpy(dev_ptr_, other.dev_ptr_, sizeof(float) * size_, cudaMemcpyDeviceToDevice));
        }
    }
    return *this;
}

// Destructor
Tensor::~Tensor() { free_storage(); }

float* Tensor::data() const {
    return (device_ == Device::GPU) ? dev_ptr_ : host_ptr_;
}

const std::vector<size_t>& Tensor::shape() const { return shape_; }
size_t Tensor::size() const { return size_; }
Device Tensor::device() const { return device_; }

void Tensor::zero_() {
    if (size_ == 0) return;
    if (device_ == Device::CPU)
        std::memset(host_ptr_, 0, sizeof(float) * size_);
    else
        CUDA_CHECK(cudaMemset(dev_ptr_, 0, sizeof(float) * size_));
}

Tensor Tensor::cpu() const {
    Tensor out(shape_, Device::CPU);
    if (size_ == 0) return out;
    if (device_ == Device::CPU)
        std::memcpy(out.host_ptr_, host_ptr_, sizeof(float) * size_);
    else
        CUDA_CHECK(cudaMemcpy(out.host_ptr_, dev_ptr_, sizeof(float) * size_, cudaMemcpyDeviceToHost));
    return out;
}

Tensor Tensor::gpu() const {
    Tensor out(shape_, Device::GPU);
    if (size_ == 0) return out;
    if (device_ == Device::GPU)
        CUDA_CHECK(cudaMemcpy(out.dev_ptr_, dev_ptr_, sizeof(float) * size_, cudaMemcpyDeviceToDevice));
    else
        CUDA_CHECK(cudaMemcpy(out.dev_ptr_, host_ptr_, sizeof(float) * size_, cudaMemcpyHostToDevice));
    return out;
}

void Tensor::fill_with(const std::vector<float>& vals) {
    assert(vals.size() == size_);
    if (size_ == 0) return;
    if (device_ == Device::CPU)
        std::memcpy(host_ptr_, vals.data(), sizeof(float) * size_);
    else
        CUDA_CHECK(cudaMemcpy(dev_ptr_, vals.data(), sizeof(float) * size_, cudaMemcpyHostToDevice));
}

std::vector<float> Tensor::to_vector() const {
    std::vector<float> out(size_);
    if (size_ == 0) return out;
    if (device_ == Device::CPU)
        std::memcpy(out.data(), host_ptr_, sizeof(float) * size_);
    else
        CUDA_CHECK(cudaMemcpy(out.data(), dev_ptr_, sizeof(float) * size_, cudaMemcpyDeviceToHost));
    return out;
}

// Internal memory management functions
void Tensor::allocate_host() {
    host_ptr_ = new float[size_]();
    dev_ptr_ = nullptr;
}

void Tensor::allocate_device() {
    dev_ptr_ = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dev_ptr_, sizeof(float) * size_));
}

void Tensor::allocate() {
    if (size_ == 0) { host_ptr_ = dev_ptr_ = nullptr; return; }
    if (device_ == Device::CPU)
        allocate_host();
    else
        allocate_device();
}

void Tensor::free_storage() {
    if (host_ptr_) {
        delete[] host_ptr_;
        host_ptr_ = nullptr;
    }
    if (dev_ptr_) {
        cudaError_t e = cudaFree(dev_ptr_);
        if (e != cudaSuccess)
            std::cerr << "cudaFree error: " << cudaGetErrorString(e) << std::endl;
        dev_ptr_ = nullptr;
    }
}

