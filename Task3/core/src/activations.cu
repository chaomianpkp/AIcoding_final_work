#include "tensor.h"
#include "activations.h"
#include "utils.h"
#include <cmath>
#include <vector>
#include <assert.h>

// ------------------- CUDA kernels -------------------
__global__ void relu_forward_kernel(const float* x, float* y, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = v > 0.0f ? v : 0.0f;
    }
}

__global__ void relu_backward_kernel(const float* x, const float* dy, float* dx, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dx[i] = (x[i] > 0.0f) ? dy[i] : 0.0f;
    }
}

__global__ void sigmoid_forward_kernel(const float* x, float* y, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = 1.0f / (1.0f + expf(-v));
    }
}

__global__ void sigmoid_backward_kernel(const float* y, const float* dy, float* dx, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float yi = y[i];
        dx[i] = dy[i] * yi * (1.0f - yi);
    }
}

// ------------------- Activation wrappers -------------------
void ReLU_forward(const Tensor& x, Tensor& y) {
    assert(x.size() == y.size());
    size_t n = x.size();
    if (n == 0) return;

    if (x.device() == Device::GPU && y.device() == Device::GPU) {
        int threads = 256;
        int blocks = (int)((n + threads - 1) / threads);
        relu_forward_kernel<<<blocks, threads>>>(x.data(), y.data(), n);
        CUDA_CHECK(cudaGetLastError(), "relu_forward kernel");
        CUDA_CHECK(cudaDeviceSynchronize(), "relu_forward sync");
        return;
    }

    // CPU path
    std::vector<float> xin = x.to_vector();
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = xin[i] > 0.0f ? xin[i] : 0.0f;
    y.fill_with(out);
}

void ReLU_backward(const Tensor& x, const Tensor& dy, Tensor& dx) {
    assert(x.size() == dy.size() && dy.size() == dx.size());
    size_t n = x.size();
    if (n == 0) return;

    if (x.device() == Device::GPU && dy.device() == Device::GPU && dx.device() == Device::GPU) {
        int threads = 256;
        int blocks = (int)((n + threads - 1) / threads);
        relu_backward_kernel<<<blocks, threads>>>(x.data(), dy.data(), dx.data(), n);
        CUDA_CHECK(cudaGetLastError(), "relu_backward kernel");
        CUDA_CHECK(cudaDeviceSynchronize(), "relu_backward sync");
        return;
    }

    // CPU path
    std::vector<float> xin = x.to_vector();
    std::vector<float> dvin = dy.to_vector();
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = xin[i] > 0.0f ? dvin[i] : 0.0f;
    dx.fill_with(out);
}

void Sigmoid_forward(const Tensor& x, Tensor& y) {
    assert(x.size() == y.size());
    size_t n = x.size();
    if (n == 0) return;

    if (x.device() == Device::GPU && y.device() == Device::GPU) {
        int threads = 256;
        int blocks = (int)((n + threads - 1) / threads);
        sigmoid_forward_kernel<<<blocks, threads>>>(x.data(), y.data(), n);
        CUDA_CHECK(cudaGetLastError(), "sigmoid_forward kernel");
        CUDA_CHECK(cudaDeviceSynchronize(), "sigmoid_forward sync");
        return;
    }

    // CPU path
    std::vector<float> xin = x.to_vector();
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = 1.0f / (1.0f + std::exp(-xin[i]));
    y.fill_with(out);
}

void Sigmoid_backward(const Tensor& y, const Tensor& dy, Tensor& dx) {
    assert(y.size() == dy.size() && dy.size() == dx.size());
    size_t n = y.size();
    if (n == 0) return;

    if (y.device() == Device::GPU && dy.device() == Device::GPU && dx.device() == Device::GPU) {
        int threads = 256;
        int blocks = (int)((n + threads - 1) / threads);
        sigmoid_backward_kernel<<<blocks, threads>>>(y.data(), dy.data(), dx.data(), n);
        CUDA_CHECK(cudaGetLastError(), "sigmoid_backward kernel");
        CUDA_CHECK(cudaDeviceSynchronize(), "sigmoid_backward sync");
        return;
    }

    // CPU path
    std::vector<float> yv = y.to_vector();
    std::vector<float> dvin = dy.to_vector();
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = dvin[i] * yv[i] * (1.0f - yv[i]);
    dx.fill_with(out);
}

