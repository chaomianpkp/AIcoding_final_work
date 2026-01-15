#include "layers.h"
#include "utils.h"
#include <cassert>
#include <cstring>
#include <cfloat>
#include <cmath>
#include <cublas_v2.h>

// ------------------- CUDA kernels -------------------

// Matrix multiplication kernel for GPU
// Computes C = A * B where A is [M, K] and B is [K, N], result C is [M, N]
__global__ void matmul_kernel(const float* A, const float* B, float* C, 
                              int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Add bias kernel: Y = Y + b (broadcast bias to each row)
__global__ void add_bias_kernel(float* Y, const float* b, int N, int Cout) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * Cout) {
        int row = idx / Cout;
        int col = idx % Cout;
        Y[idx] += b[col];
    }
}

// Matrix multiplication with transpose: C = A * B^T
// A is [M, K], B is [N, K], result C is [M, N]
__global__ void matmul_transpose_b_kernel(const float* A, const float* B, float* C,
                                          int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[col * K + k];  // B^T access
        }
        C[row * N + col] = sum;
    }
}

// Matrix multiplication with transpose: C = A^T * B
// A is [K, M], B is [K, N], result C is [M, N]
__global__ void matmul_transpose_a_kernel(const float* A, const float* B, float* C,
                                          int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[k * M + row] * B[k * N + col];  // A^T access
        }
        C[row * N + col] = sum;
    }
}

// Sum reduction kernel for bias gradient: db = sum(dY, axis=0)
__global__ void sum_columns_kernel(const float* dY, float* db, int N, int Cout) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < Cout) {
        float sum = 0.0f;
        for (int row = 0; row < N; ++row) {
            sum += dY[row * Cout + col];
        }
        db[col] = sum;
    }
}

// ------------------- Helper functions -------------------

static void cpu_matmul(const float* A, const float* B, float* C, 
                       int M, int N, int K) {
    // C = A * B, where A is [M, K], B is [K, N], C is [M, N]
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

static void cpu_matmul_transpose_b(const float* A, const float* B, float* C,
                                   int M, int N, int K) {
    // C = A * B^T, where A is [M, K], B is [N, K], C is [M, N]
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] = sum;
        }
    }
}

static void cpu_matmul_transpose_a(const float* A, const float* B, float* C,
                                   int M, int N, int K) {
    // C = A^T * B, where A is [K, M], B is [K, N], C is [M, N]
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[k * M + i] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ------------------- Linear Layer Implementation -------------------

void Linear_forward(const Tensor& X, const Tensor& W, const Tensor& b, Tensor& Y) {
    // Check shapes
    assert(X.shape().size() == 2);
    assert(W.shape().size() == 2);
    assert(b.shape().size() == 1);
    assert(Y.shape().size() == 2);
    
    size_t N = X.shape()[0];      // batch size
    size_t Cin = X.shape()[1];     // input channels
    size_t Cout = W.shape()[0];    // output channels
    size_t W_Cin = W.shape()[1];   // weight input channels
    
    assert(Cin == W_Cin);
    assert(Y.shape()[0] == N && Y.shape()[1] == Cout);
    assert(b.shape()[0] == Cout);
    
    // Forward: Y = X * W^T + b
    // X: [N, Cin], W: [Cout, Cin], Y: [N, Cout], b: [Cout]
    
    if (X.device() == Device::GPU && W.device() == Device::GPU && 
        b.device() == Device::GPU && Y.device() == Device::GPU) {
        // GPU path using cuBLAS
        cublasHandle_t handle;
        cublasCreate(&handle);
        
        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        // Y = X * W^T (using cublasSgemm)
        //  Y^T[Cout, N] = W^T[Cout, Cin] * X[Cin, N] = W[Cout, Cin] (in row-major) * X^T[Cin, N]
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    Cout, N, Cin,
                    &alpha,
                    W.data(), Cin,    // A = W^T [Cout, Cin] (W is [Cin, Cout] col-major), lda = Cin
                    X.data(), Cin,    // B = X [Cin, N] col-major, ldb = Cin
                    &beta,
                    Y.data(), Cout);  // C = Y^T [Cout, N] col-major, ldc = Cout
        
        // Add bias: Y = Y + b (broadcast)
        dim3 threads(256);
        dim3 blocks((N * Cout + threads.x - 1) / threads.x);
        add_bias_kernel<<<blocks, threads>>>(Y.data(), b.data(), N, Cout);
        CUDA_CHECK(cudaGetLastError(), "add_bias_kernel");
        CUDA_CHECK(cudaDeviceSynchronize(), "add_bias sync");
        
        cublasDestroy(handle);
        return;
    }
    
    // CPU path
    std::vector<float> X_vec = X.to_vector();
    std::vector<float> W_vec = W.to_vector();
    std::vector<float> b_vec = b.to_vector();
    std::vector<float> Y_vec(N * Cout, 0.0f);
    
    // Y = X * W^T
    cpu_matmul_transpose_b(X_vec.data(), W_vec.data(), Y_vec.data(), 
                           N, Cout, Cin);
    
    // Add bias
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < Cout; ++j) {
            Y_vec[i * Cout + j] += b_vec[j];
        }
    }
    
    Y.fill_with(Y_vec);
}

void Linear_backward(const Tensor& X, const Tensor& W, const Tensor& dY,
                     Tensor& dX, Tensor& dW, Tensor& db) {
    // Check shapes
    assert(X.shape().size() == 2);
    assert(W.shape().size() == 2);
    assert(dY.shape().size() == 2);
    assert(dX.shape().size() == 2);
    assert(dW.shape().size() == 2);
    assert(db.shape().size() == 1);
    
    size_t N = X.shape()[0];
    size_t Cin = X.shape()[1];
    size_t Cout = W.shape()[0];
    
    assert(dY.shape()[0] == N && dY.shape()[1] == Cout);
    assert(dX.shape()[0] == N && dX.shape()[1] == Cin);
    assert(dW.shape()[0] == Cout && dW.shape()[1] == Cin);
    assert(db.shape()[0] == Cout);
    
    // Backward:
    // dX = dY * W (dY: [N, Cout], W: [Cout, Cin], dX: [N, Cin])
    // dW = dY^T * X (dY: [N, Cout], X: [N, Cin], dW: [Cout, Cin])
    // db = sum(dY, axis=0) (dY: [N, Cout], db: [Cout])
    
    if (X.device() == Device::GPU && W.device() == Device::GPU &&
        dY.device() == Device::GPU && dX.device() == Device::GPU &&
        dW.device() == Device::GPU && db.device() == Device::GPU) {
        // GPU path using cuBLAS
        cublasHandle_t handle;
        cublasCreate(&handle);
        
        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        // dX = dY * W
        // Row-major: dX[N, Cin] = dY[N, Cout] * W[Cout, Cin]
        // dX^T[Cin, N] = W[Cin, Cout] * dY[Cout, N]
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    Cin, N, Cout,
                    &alpha,
                    W.data(), Cin,    // A = W[Cin, Cout] col-major, op(A)=N gives W[Cin, Cout]
                    dY.data(), Cout,  // B = dY[Cout, N] col-major, op(B)=N gives dY[Cout, N]
                    &beta,
                    dX.data(), Cin);  // C = dX^T[Cin, N] col-major, ldc = Cin
        
        // dW = dY^T * X
        // Row-major: dW[Cout, Cin] = dY^T[Cout, N] * X[N, Cin]
        // In cuBLAS: dW^T[Cin, Cout] = X^T[Cin, N] * dY[Cout, N]
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    Cin, Cout, N,
                    &alpha,
                    X.data(), Cin,    // A = X[Cin, N] col-major, op(A)=N gives X[Cin, N]
                    dY.data(), Cout,  // B = dY[Cout, N] col-major, op(B)=T gives dY^T[N, Cout]
                    &beta,
                    dW.data(), Cin);  // C = dW^T[Cin, Cout] col-major, ldc = Cin
        
        // db = sum(dY, axis=0)
        dim3 threads(256);
        dim3 blocks((Cout + threads.x - 1) / threads.x);
        sum_columns_kernel<<<blocks, threads>>>(dY.data(), db.data(), N, Cout);
        CUDA_CHECK(cudaGetLastError(), "sum_columns_kernel");
        CUDA_CHECK(cudaDeviceSynchronize(), "sum_columns sync");
        
        cublasDestroy(handle);
        return;
    }
    
    // CPU path
    std::vector<float> X_vec = X.to_vector();
    std::vector<float> W_vec = W.to_vector();
    std::vector<float> dY_vec = dY.to_vector();
    std::vector<float> dX_vec(N * Cin, 0.0f);
    std::vector<float> dW_vec(Cout * Cin, 0.0f);
    std::vector<float> db_vec(Cout, 0.0f);
    
    // dX = dY * W
    cpu_matmul(dY_vec.data(), W_vec.data(), dX_vec.data(), N, Cin, Cout);
    
    // dW = dY^T * X
    cpu_matmul_transpose_a(dY_vec.data(), X_vec.data(), dW_vec.data(), 
                           Cout, Cin, N);
    
    // db = sum(dY, axis=0)
    for (size_t j = 0; j < Cout; ++j) {
        float sum = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            sum += dY_vec[i * Cout + j];
        }
        db_vec[j] = sum;
    }
    
    dX.fill_with(dX_vec);
    dW.fill_with(dW_vec);
    db.fill_with(db_vec);
}

// ------------------- Convolution Layer Implementation -------------------

// CUDA kernel for im2col
__global__ void im2col_kernel(const float* X, float* Col,
                               int N, int Cin, int H, int W,
                               int kernel_size, int pad, int stride,
                               int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H_out * W_out * Cin * kernel_size * kernel_size;
    
    if (idx < total) {
        int n = idx / (H_out * W_out * Cin * kernel_size * kernel_size);
        int h_out = (idx / (W_out * Cin * kernel_size * kernel_size)) % H_out;
        int w_out = (idx / (Cin * kernel_size * kernel_size)) % W_out;
        int c = (idx / (kernel_size * kernel_size)) % Cin;
        int kh = (idx / kernel_size) % kernel_size;
        int kw = idx % kernel_size;
        
        int h_in = h_out * stride + kh - pad;
        int w_in = w_out * stride + kw - pad;
        
        int col_row = n * H_out * W_out + h_out * W_out + w_out;
        int col_col = c * kernel_size * kernel_size + kh * kernel_size + kw;
        
        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
            int img_idx = n * Cin * H * W + c * H * W + h_in * W + w_in;
            Col[col_row * (Cin * kernel_size * kernel_size) + col_col] = X[img_idx];
        } else {
            Col[col_row * (Cin * kernel_size * kernel_size) + col_col] = 0.0f;
        }
    }
}

// CUDA kernel for col2im
__global__ void col2im_kernel(const float* Col, float* X,
                               int N, int Cin, int H, int W,
                               int kernel_size, int pad, int stride,
                               int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cin * H * W;
    
    if (idx < total) {
        int n = idx / (Cin * H * W);
        int c = (idx / (H * W)) % Cin;
        int h = (idx / W) % H;
        int w = idx % W;
        
        float sum = 0.0f;
        
        // Iterate over all positions in output that could contribute to this input position
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_out = (h + pad - kh + stride - 1) / stride;
                int w_out = (w + pad - kw + stride - 1) / stride;
                
                if (h_out >= 0 && h_out < H_out && w_out >= 0 && w_out < W_out &&
                    (h + pad - kh) % stride == 0 && (w + pad - kw) % stride == 0) {
                    
                    int col_row = n * H_out * W_out + h_out * W_out + w_out;
                    int col_col = c * kernel_size * kernel_size + kh * kernel_size + kw;
                    
                    sum += Col[col_row * (Cin * kernel_size * kernel_size) + col_col];
                }
            }
        }
        
        X[idx] = sum;
    }
}

void im2col(const Tensor& X, Tensor& Col, int kernel_size, int pad, int stride) {
    assert(X.shape().size() == 4);
    
    int N = X.shape()[0];
    int Cin = X.shape()[1];
    int H = X.shape()[2];
    int W = X.shape()[3];
    
    int H_out = (H + 2 * pad - kernel_size) / stride + 1;
    int W_out = (W + 2 * pad - kernel_size) / stride + 1;
    
    assert(Col.shape().size() == 2);
    assert(Col.shape()[0] == N * H_out * W_out);
    assert(Col.shape()[1] == Cin * kernel_size * kernel_size);
    
    if (X.device() == Device::GPU && Col.device() == Device::GPU) {
        int total = N * H_out * W_out * Cin * kernel_size * kernel_size;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        
        im2col_kernel<<<blocks, threads>>>(
            X.data(), Col.data(), N, Cin, H, W, kernel_size, pad, stride, H_out, W_out);
        CUDA_CHECK(cudaGetLastError(), "im2col_kernel");
        CUDA_CHECK(cudaDeviceSynchronize(), "im2col sync");
        return;
    }
    
    // CPU path
    std::vector<float> X_vec = X.to_vector();
    std::vector<float> Col_vec(N * H_out * W_out * Cin * kernel_size * kernel_size, 0.0f);
    
    for (int n = 0; n < N; ++n) {
        for (int h_out = 0; h_out < H_out; ++h_out) {
            for (int w_out = 0; w_out < W_out; ++w_out) {
                for (int c = 0; c < Cin; ++c) {
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int h_in = h_out * stride + kh - pad;
                            int w_in = w_out * stride + kw - pad;
                            
                            int col_row = n * H_out * W_out + h_out * W_out + w_out;
                            int col_col = c * kernel_size * kernel_size + kh * kernel_size + kw;
                            
                            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                int img_idx = n * Cin * H * W + c * H * W + h_in * W + w_in;
                                Col_vec[col_row * (Cin * kernel_size * kernel_size) + col_col] = X_vec[img_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    
    Col.fill_with(Col_vec);
}

void col2im(const Tensor& Col, Tensor& X, int N, int Cin, int H, int W,
            int kernel_size, int pad, int stride) {
    assert(Col.shape().size() == 2);
    assert(X.shape().size() == 4);
    assert(X.shape()[0] == N && X.shape()[1] == Cin && 
           X.shape()[2] == H && X.shape()[3] == W);
    
    int H_out = (H + 2 * pad - kernel_size) / stride + 1;
    int W_out = (W + 2 * pad - kernel_size) / stride + 1;
    
    assert(Col.shape()[0] == N * H_out * W_out);
    assert(Col.shape()[1] == Cin * kernel_size * kernel_size);
    
    if (Col.device() == Device::GPU && X.device() == Device::GPU) {
        X.zero_();  // Initialize to zero
        
        int total = N * Cin * H * W;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        
        col2im_kernel<<<blocks, threads>>>(
            Col.data(), X.data(), N, Cin, H, W, kernel_size, pad, stride, H_out, W_out);
        CUDA_CHECK(cudaGetLastError(), "col2im_kernel");
        CUDA_CHECK(cudaDeviceSynchronize(), "col2im sync");
        return;
    }
    
    // CPU path
    std::vector<float> Col_vec = Col.to_vector();
    std::vector<float> X_vec(N * Cin * H * W, 0.0f);
    
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < Cin; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    float sum = 0.0f;
                    
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int h_out = (h + pad - kh + stride - 1) / stride;
                            int w_out = (w + pad - kw + stride - 1) / stride;
                            
                            if (h_out >= 0 && h_out < H_out && w_out >= 0 && w_out < W_out &&
                                (h + pad - kh) % stride == 0 && (w + pad - kw) % stride == 0) {
                                
                                int col_row = n * H_out * W_out + h_out * W_out + w_out;
                                int col_col = c * kernel_size * kernel_size + kh * kernel_size + kw;
                                
                                sum += Col_vec[col_row * (Cin * kernel_size * kernel_size) + col_col];
                            }
                        }
                    }
                    
                    int img_idx = n * Cin * H * W + c * H * W + h * W + w;
                    X_vec[img_idx] = sum;
                }
            }
        }
    }
    
    X.fill_with(X_vec);
}

void Conv2d_forward(const Tensor& X, const Tensor& W, const Tensor& b, Tensor& Y) {
    // Check shapes
    assert(X.shape().size() == 4);
    assert(W.shape().size() == 4);
    assert(b.shape().size() == 1);
    assert(Y.shape().size() == 4);
    
    int N = X.shape()[0];
    int Cin = X.shape()[1];
    int H = X.shape()[2];
    int W_in = X.shape()[3];
    
    int Cout = W.shape()[0];
    int W_Cin = W.shape()[1];
    int K = W.shape()[2];
    assert(W.shape()[3] == K && K == 3);
    
    assert(Cin == W_Cin);
    assert(Y.shape()[0] == N && Y.shape()[1] == Cout && 
           Y.shape()[2] == H && Y.shape()[3] == W_in);
    assert(b.shape()[0] == Cout);
    
    int pad = 1;
    int stride = 1;
    int H_out = H;  // With pad=1, stride=1, 3x3 kernel, output size = input size
    int W_out = W_in;
    
    // Step 1: im2col - convert input to column matrix
    // Col: [N*H*W, Cin*K*K]
    Tensor Col({static_cast<size_t>(N * H_out * W_out), static_cast<size_t>(Cin * K * K)}, X.device());
    im2col(X, Col, K, pad, stride);

    // Step 2: Reshape W to [Cout, Cin*K*K]
    // W_col: [Cout, Cin*K*K]
    Tensor W_col({static_cast<size_t>(Cout), static_cast<size_t>(Cin * K * K)}, W.device());
    if (W.device() == Device::GPU) {
        // Copy and reshape on GPU
        std::vector<float> W_vec = W.to_vector();
        std::vector<float> W_col_vec(Cout * Cin * K * K);
        for (int i = 0; i < Cout; ++i) {
            for (int j = 0; j < Cin * K * K; ++j) {
                W_col_vec[i * (Cin * K * K) + j] = W_vec[i * (Cin * K * K) + j];
            }
        }
        W_col.fill_with(W_col_vec);
    } else {
        std::vector<float> W_vec = W.to_vector();
        W_col.fill_with(W_vec);
    }
    
    // Step 3: Matrix multiplication: Y_col = Col * W_col^T
    // Col: [N*H*W, Cin*K*K], W_col: [Cout, Cin*K*K]
    // Y_col: [N*H*W, Cout]
    Tensor Y_col({static_cast<size_t>(N * H_out * W_out), static_cast<size_t>(Cout)}, X.device());
    
    if (Col.device() == Device::GPU && W_col.device() == Device::GPU && 
        Y_col.device() == Device::GPU) {
        cublasHandle_t handle;
        cublasCreate(&handle);
        
        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        // Y_col = Col * W_col^T
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    N * H_out * W_out, Cout, Cin * K * K,
                    &alpha,
                    Col.data(), N * H_out * W_out,
                    W_col.data(), Cout,
                    &beta,
                    Y_col.data(), N * H_out * W_out);
        
        cublasDestroy(handle);
    } else {
        // CPU path
        std::vector<float> Col_vec = Col.to_vector();
        std::vector<float> W_col_vec = W_col.to_vector();
        std::vector<float> Y_col_vec(N * H_out * W_out * Cout, 0.0f);
        
        cpu_matmul_transpose_b(Col_vec.data(), W_col_vec.data(), Y_col_vec.data(),
                               N * H_out * W_out, Cout, Cin * K * K);
        Y_col.fill_with(Y_col_vec);
    }
    
    // Step 4: Add bias and reshape to [N, Cout, H, W]
    std::vector<float> Y_col_vec = Y_col.to_vector();
    std::vector<float> Y_vec(N * Cout * H_out * W_out, 0.0f);
    std::vector<float> b_vec = b.to_vector();
    
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H_out; ++h) {
            for (int w = 0; w < W_out; ++w) {
                for (int c = 0; c < Cout; ++c) {
                    int col_idx = (n * H_out * W_out + h * W_out + w) * Cout + c;
                    int img_idx = n * Cout * H_out * W_out + c * H_out * W_out + h * W_out + w;
                    Y_vec[img_idx] = Y_col_vec[col_idx] + b_vec[c];
                }
            }
        }
    }
    
    Y.fill_with(Y_vec);
}

void Conv2d_backward(const Tensor& X, const Tensor& W, const Tensor& dY,
                     Tensor& dX, Tensor& dW, Tensor& db) {
    // Check shapes
    assert(X.shape().size() == 4);
    assert(W.shape().size() == 4);
    assert(dY.shape().size() == 4);
    assert(dX.shape().size() == 4);
    assert(dW.shape().size() == 4);
    assert(db.shape().size() == 1);
    
    int N = X.shape()[0];
    int Cin = X.shape()[1];
    int H = X.shape()[2];
    int W_in = X.shape()[3];
    
    int Cout = W.shape()[0];
    int K = W.shape()[2];
    
    assert(dY.shape()[0] == N && dY.shape()[1] == Cout && 
           dY.shape()[2] == H && dY.shape()[3] == W_in);
    assert(dX.shape() == X.shape());
    assert(dW.shape() == W.shape());
    assert(db.shape()[0] == Cout);
    
    int pad = 1;
    int stride = 1;
    int H_out = H;
    int W_out = W_in;
    
    // Step 1: Reshape dY to [N*H*W, Cout]
    std::vector<float> dY_vec = dY.to_vector();
    std::vector<float> dY_col_vec(N * H_out * W_out * Cout);
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H_out; ++h) {
            for (int w = 0; w < W_out; ++w) {
                for (int c = 0; c < Cout; ++c) {
                    int img_idx = n * Cout * H_out * W_out + c * H_out * W_out + h * W_out + w;
                    int col_idx = (n * H_out * W_out + h * W_out + w) * Cout + c;
                    dY_col_vec[col_idx] = dY_vec[img_idx];
                }
            }
        }
    }
    Tensor dY_col({static_cast<size_t>(N * H_out * W_out), static_cast<size_t>(Cout)}, dY.device());
    dY_col.fill_with(dY_col_vec);
    
    // Step 2: im2col on X for dX computation
    Tensor Col({static_cast<size_t>(N * H_out * W_out), static_cast<size_t>(Cin * K * K)}, X.device());
    im2col(X, Col, K, pad, stride);
    
    // Step 3: Compute dW = dY_col^T * Col
    // dY_col: [N*H*W, Cout], Col: [N*H*W, Cin*K*K]
    // dW: [Cout, Cin*K*K]
    Tensor dW_col({static_cast<size_t>(Cout), static_cast<size_t>(Cin * K * K)}, W.device());
    
    if (dY_col.device() == Device::GPU && Col.device() == Device::GPU &&
        dW_col.device() == Device::GPU) {
        cublasHandle_t handle;
        cublasCreate(&handle);
        
        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        // dW_col = dY_col^T * Col
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    Cout, Cin * K * K, N * H_out * W_out,
                    &alpha,
                    dY_col.data(), N * H_out * W_out,
                    Col.data(), N * H_out * W_out,
                    &beta,
                    dW_col.data(), Cout);
        
        cublasDestroy(handle);
    } else {
        std::vector<float> dY_col_vec = dY_col.to_vector();
        std::vector<float> Col_vec = Col.to_vector();
        std::vector<float> dW_col_vec(Cout * Cin * K * K, 0.0f);
        
        cpu_matmul_transpose_a(dY_col_vec.data(), Col_vec.data(), dW_col_vec.data(),
                               Cout, Cin * K * K, N * H_out * W_out);
        dW_col.fill_with(dW_col_vec);
    }
    
    // Reshape dW_col to [Cout, Cin, K, K]
    std::vector<float> dW_col_vec = dW_col.to_vector();
    std::vector<float> dW_vec(Cout * Cin * K * K);
    std::copy(dW_col_vec.begin(), dW_col_vec.end(), dW_vec.begin());
    dW.fill_with(dW_vec);
    
    // Step 4: Reshape W to [Cout, Cin*K*K]
    Tensor W_col({static_cast<size_t>(Cout), static_cast<size_t>(Cin * K * K)}, W.device());
    std::vector<float> W_vec = W.to_vector();
    W_col.fill_with(W_vec);
    
    // Step 5: Compute dCol = dY_col * W_col
    // dY_col: [N*H*W, Cout], W_col: [Cout, Cin*K*K]
    // dCol: [N*H*W, Cin*K*K]
    Tensor dCol({static_cast<size_t>(N * H_out * W_out), static_cast<size_t>(Cin * K * K)}, X.device());
    
    if (dY_col.device() == Device::GPU && W_col.device() == Device::GPU &&
        dCol.device() == Device::GPU) {
        cublasHandle_t handle;
        cublasCreate(&handle);
        
        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        // dCol = dY_col * W_col
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N * H_out * W_out, Cin * K * K, Cout,
                    &alpha,
                    dY_col.data(), N * H_out * W_out,
                    W_col.data(), Cout,
                    &beta,
                    dCol.data(), N * H_out * W_out);
        
        cublasDestroy(handle);
    } else {
        std::vector<float> dY_col_vec = dY_col.to_vector();
        std::vector<float> W_col_vec = W_col.to_vector();
        std::vector<float> dCol_vec(N * H_out * W_out * Cin * K * K, 0.0f);
        
        cpu_matmul(dY_col_vec.data(), W_col_vec.data(), dCol_vec.data(),
                   N * H_out * W_out, Cin * K * K, Cout);
        dCol.fill_with(dCol_vec);
    }
    
    // Step 6: col2im to convert dCol back to dX
    col2im(dCol, dX, N, Cin, H, W_in, K, pad, stride);
    
    // Step 7: Compute db = sum(dY, axis=(0,2,3))
    std::vector<float> db_vec(Cout, 0.0f);
    for (int c = 0; c < Cout; ++c) {
        float sum = 0.0f;
        for (int n = 0; n < N; ++n) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    int idx = n * Cout * H_out * W_out + c * H_out * W_out + h * W_out + w;
                    sum += dY_vec[idx];
                }
            }
        }
        db_vec[c] = sum;
    }
    db.fill_with(db_vec);
}

// ------------------- Pooling Layer -------------------

__global__ void maxpool2d_forward_kernel(const float* X, float* Y,
                                         int N, int C, int H, int W,
                                         int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c = (idx / (W_out * H_out)) % C;
    int n = idx / (W_out * H_out * C);

    int h_start = h_out * 2;
    int w_start = w_out * 2;

    float max_val = -FLT_MAX;
    int base = ((n * C + c) * H) * W;
    for (int kh = 0; kh < 2; ++kh) {
        for (int kw = 0; kw < 2; ++kw) {
            int h = h_start + kh;
            int w = w_start + kw;
            int idx_in = base + h * W + w;
            float val = X[idx_in];
            if (val > max_val) {
                max_val = val;
            }
        }
    }
    Y[idx] = max_val;
}

__global__ void maxpool2d_backward_kernel(const float* X, const float* Y,
                                          const float* dY, float* dX,
                                          int N, int C, int H, int W,
                                          int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c = (idx / (W_out * H_out)) % C;
    int n = idx / (W_out * H_out * C);

    int h_start = h_out * 2;
    int w_start = w_out * 2;
    float grad = dY[idx];
    float max_val = Y[idx];

    int base = ((n * C + c) * H) * W;
    for (int kh = 0; kh < 2; ++kh) {
        for (int kw = 0; kw < 2; ++kw) {
            int h = h_start + kh;
            int w = w_start + kw;
            int idx_in = base + h * W + w;
            if (X[idx_in] == max_val) {
                atomicAdd(&dX[idx_in], grad);
            }
        }
    }
}

void MaxPool2d_forward(const Tensor& X, Tensor& Y) {
    assert(X.shape().size() == 4);
    assert(Y.shape().size() == 4);

    int N = X.shape()[0];
    int C = X.shape()[1];
    int H = X.shape()[2];
    int W = X.shape()[3];
    assert(H % 2 == 0 && W % 2 == 0);

    int H_out = H / 2;
    int W_out = W / 2;
    assert(Y.shape()[0] == N && Y.shape()[1] == C &&
           Y.shape()[2] == (size_t)H_out && Y.shape()[3] == (size_t)W_out);

    if (X.device() == Device::GPU && Y.device() == Device::GPU) {
        int total = N * C * H_out * W_out;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;

        maxpool2d_forward_kernel<<<blocks, threads>>>(X.data(), Y.data(),
                                                      N, C, H, W, H_out, W_out);
        CUDA_CHECK(cudaGetLastError(), "maxpool2d_forward_kernel");
        CUDA_CHECK(cudaDeviceSynchronize(), "maxpool2d forward sync");
        return;
    }

    // CPU path
    std::vector<float> X_vec = X.to_vector();
    std::vector<float> Y_vec(N * C * H_out * W_out, 0.0f);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h_out = 0; h_out < H_out; ++h_out) {
                for (int w_out = 0; w_out < W_out; ++w_out) {
                    float max_val = -FLT_MAX;
                    for (int kh = 0; kh < 2; ++kh) {
                        for (int kw = 0; kw < 2; ++kw) {
                            int h = h_out * 2 + kh;
                            int w = w_out * 2 + kw;
                            int idx_in = (((n * C + c) * H) + h) * W + w;
                            float val = X_vec[idx_in];
                            if (val > max_val) max_val = val;
                        }
                    }
                    int idx_out = (((n * C + c) * H_out) + h_out) * W_out + w_out;
                    Y_vec[idx_out] = max_val;
                }
            }
        }
    }
    Y.fill_with(Y_vec);
}

void MaxPool2d_backward(const Tensor& X, const Tensor& Y, const Tensor& dY, Tensor& dX) {
    assert(X.shape().size() == 4);
    assert(Y.shape().size() == 4);
    assert(dY.shape().size() == 4);
    assert(dX.shape().size() == 4);

    int N = X.shape()[0];
    int C = X.shape()[1];
    int H = X.shape()[2];
    int W = X.shape()[3];
    int H_out = Y.shape()[2];
    int W_out = Y.shape()[3];

    assert(H_out * 2 == H && W_out * 2 == W);
    assert(dY.shape()[0] == N && dY.shape()[1] == C &&
           dY.shape()[2] == (size_t)H_out && dY.shape()[3] == (size_t)W_out);
    assert(dX.shape() == X.shape());

    dX.zero_();

    if (X.device() == Device::GPU && Y.device() == Device::GPU &&
        dY.device() == Device::GPU && dX.device() == Device::GPU) {
        int total = N * C * H_out * W_out;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;

        maxpool2d_backward_kernel<<<blocks, threads>>>(X.data(), Y.data(), dY.data(), dX.data(),
                                                       N, C, H, W, H_out, W_out);
        CUDA_CHECK(cudaGetLastError(), "maxpool2d_backward_kernel");
        CUDA_CHECK(cudaDeviceSynchronize(), "maxpool2d backward sync");
        return;
    }

    // CPU path
    std::vector<float> X_vec = X.to_vector();
    std::vector<float> Y_vec = Y.to_vector();
    std::vector<float> dY_vec = dY.to_vector();
    std::vector<float> dX_vec(N * C * H * W, 0.0f);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h_out = 0; h_out < H_out; ++h_out) {
                for (int w_out = 0; w_out < W_out; ++w_out) {
                    int idx_out = (((n * C + c) * H_out) + h_out) * W_out + w_out;
                    float max_val = Y_vec[idx_out];
                    float grad = dY_vec[idx_out];

                    for (int kh = 0; kh < 2; ++kh) {
                        for (int kw = 0; kw < 2; ++kw) {
                            int h = h_out * 2 + kh;
                            int w = w_out * 2 + kw;
                            int idx_in = (((n * C + c) * H) + h) * W + w;
                            if (X_vec[idx_in] == max_val) {
                                dX_vec[idx_in] += grad;
                            }
                        }
                    }
                }
            }
        }
    }
    dX.fill_with(dX_vec);
}

// ------------------- Softmax Layer -------------------

void Softmax_forward(const Tensor& X, Tensor& Y) {
    assert(X.shape().size() == 2);
    assert(Y.shape().size() == 2);
    assert(Y.shape()[0] == X.shape()[0] && Y.shape()[1] == X.shape()[1]);

    size_t N = X.shape()[0];
    size_t C = X.shape()[1];

    if (N == 0 || C == 0) {
        Y.zero_();
        return;
    }

    std::vector<float> X_vec = X.to_vector();
    std::vector<float> Y_vec(N * C, 0.0f);

    for (size_t n = 0; n < N; ++n) {
        size_t row_offset = n * C;
        float max_val = -FLT_MAX;
        for (size_t c = 0; c < C; ++c) {
            max_val = std::max(max_val, X_vec[row_offset + c]);
        }

        float sum = 0.0f;
        for (size_t c = 0; c < C; ++c) {
            float val = std::exp(X_vec[row_offset + c] - max_val);
            Y_vec[row_offset + c] = val;
            sum += val;
        }

        if (sum > 0.0f) {
            float inv_sum = 1.0f / sum;
            for (size_t c = 0; c < C; ++c) {
                Y_vec[row_offset + c] *= inv_sum;
            }
        }
    }

    Y.fill_with(Y_vec);
}

float CrossEntropyLoss_forward(const Tensor& probs, const Tensor& labels) {
    assert(probs.shape().size() == 2);
    assert(labels.shape().size() == 1);
    size_t N = probs.shape()[0];
    size_t C = probs.shape()[1];
    assert(labels.shape()[0] == N);

    if (N == 0 || C == 0) return 0.0f;

    std::vector<float> probs_vec = probs.to_vector();
    std::vector<float> labels_vec = labels.to_vector();

    const float eps = 1e-12f;
    float loss = 0.0f;

    for (size_t n = 0; n < N; ++n) {
        int label = static_cast<int>(labels_vec[n]);
        assert(label >= 0 && label < static_cast<int>(C));
        float p = probs_vec[n * C + label];
        loss -= std::log(std::max(p, eps));
    }

    loss /= static_cast<float>(N);
    return loss;
}

void CrossEntropyLoss_backward(const Tensor& probs, const Tensor& labels, Tensor& dLogits) {
    assert(probs.shape().size() == 2);
    assert(labels.shape().size() == 1);
    assert(dLogits.shape().size() == 2);

    size_t N = probs.shape()[0];
    size_t C = probs.shape()[1];
    assert(labels.shape()[0] == N);
    assert(dLogits.shape()[0] == N && dLogits.shape()[1] == C);

    if (N == 0 || C == 0) {
        dLogits.zero_();
        return;
    }

    std::vector<float> probs_vec = probs.to_vector();
    std::vector<float> labels_vec = labels.to_vector();
    std::vector<float> grad_vec = probs_vec;

    for (size_t n = 0; n < N; ++n) {
        int label = static_cast<int>(labels_vec[n]);
        assert(label >= 0 && label < static_cast<int>(C));
        grad_vec[n * C + label] -= 1.0f;
    }

    float invN = 1.0f / static_cast<float>(N);
    for (float &g : grad_vec) g *= invN;

    dLogits.fill_with(grad_vec);
}

