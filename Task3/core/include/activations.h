#pragma once
#include "tensor.h"

// Forward declarations for activation functions
void ReLU_forward(const Tensor& x, Tensor& y);
void ReLU_backward(const Tensor& x, const Tensor& dy, Tensor& dx);

void Sigmoid_forward(const Tensor& x, Tensor& y);
void Sigmoid_backward(const Tensor& y, const Tensor& dy, Tensor& dx);

