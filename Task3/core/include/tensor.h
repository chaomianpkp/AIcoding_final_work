#pragma once
#include <vector>
#include <cstddef>

enum class Device { CPU, GPU };

class Tensor {
public:
    Tensor();
    Tensor(const std::vector<size_t>& shape, Device device = Device::CPU);
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    ~Tensor();

    float* data() const;
    const std::vector<size_t>& shape() const;
    size_t size() const;
    Device device() const;

    void zero_();
    void fill_with(const std::vector<float>& vals);
    std::vector<float> to_vector() const;

    Tensor cpu() const;
    Tensor gpu() const;

private:
    std::vector<size_t> shape_;
    size_t size_;
    Device device_;
    float* host_ptr_;
    float* dev_ptr_;

    void allocate();
    void allocate_host();
    void allocate_device();
    void free_storage();
};

