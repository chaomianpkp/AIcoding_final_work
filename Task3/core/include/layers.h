#pragma once
#include "tensor.h"

// Fully Connected Layer (Linear Layer)
// Forward: Y = W * X + b
//   X: [N, Cin] - input tensor
//   W: [Cout, Cin] - weight matrix
//   b: [Cout] - bias vector
//   Y: [N, Cout] - output tensor
void Linear_forward(const Tensor& X, const Tensor& W, const Tensor& b, Tensor& Y);

// Backward: compute gradients
//   dY: [N, Cout] - gradient w.r.t. output
//   dX: [N, Cin] - gradient w.r.t. input (output)
//   dW: [Cout, Cin] - gradient w.r.t. weights (output)
//   db: [Cout] - gradient w.r.t. bias (output)
void Linear_backward(const Tensor& X, const Tensor& W, const Tensor& dY, 
                     Tensor& dX, Tensor& dW, Tensor& db);

// ------------------- Convolution Layer -------------------

// im2col: Convert image to column matrix for convolution
//   Input: X [N, Cin, H, W]
//   Output: Col [N * H * W, Cin * K * K] where K = kernel_size (3)
//   Padding: pad = 1, Stride: stride = 1
void im2col(const Tensor& X, Tensor& Col, int kernel_size, int pad, int stride);

// col2im: Convert column matrix back to image (for backward pass)
//   Input: Col [N * H * W, Cin * K * K]
//   Output: X [N, Cin, H, W]
//   Padding: pad = 1, Stride: stride = 1
void col2im(const Tensor& Col, Tensor& X, int N, int Cin, int H, int W, 
            int kernel_size, int pad, int stride);

// Conv2d forward: 2D Convolution
//   X: [N, Cin, H, W] - input tensor
//   W: [Cout, Cin, 3, 3] - weight filters (3x3 kernels)
//   b: [Cout] - bias vector
//   Y: [N, Cout, H, W] - output tensor
//   Padding: 1, Stride: 1
void Conv2d_forward(const Tensor& X, const Tensor& W, const Tensor& b, Tensor& Y);

// Conv2d backward: compute gradients
//   dY: [N, Cout, H, W] - gradient w.r.t. output
//   dX: [N, Cin, H, W] - gradient w.r.t. input (output)
//   dW: [Cout, Cin, 3, 3] - gradient w.r.t. weights (output)
//   db: [Cout] - gradient w.r.t. bias (output)
void Conv2d_backward(const Tensor& X, const Tensor& W, const Tensor& dY,
                     Tensor& dX, Tensor& dW, Tensor& db);

// ------------------- Pooling Layer -------------------

// Max Pooling forward (kernel=2, stride=2)
//   X: [N, C, H, W]
//   Y: [N, C, H/2, W/2]
void MaxPool2d_forward(const Tensor& X, Tensor& Y);

// Max Pooling backward
//   X: [N, C, H, W], Y: [N, C, H/2, W/2]
//   dY: gradient wrt Y, dX: output gradients wrt X
void MaxPool2d_backward(const Tensor& X, const Tensor& Y, const Tensor& dY, Tensor& dX);

// ------------------- Softmax Layer -------------------

// Softmax forward (stable): input [N, C], output [N, C]
void Softmax_forward(const Tensor& X, Tensor& Y);

// Cross Entropy Loss (expects Softmax probabilities)
// labels Tensor shape [N] storing class indices (0..C-1)
float CrossEntropyLoss_forward(const Tensor& probs, const Tensor& labels);
void CrossEntropyLoss_backward(const Tensor& probs, const Tensor& labels, Tensor& dLogits);

