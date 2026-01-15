#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <vector>

inline void CUDA_CHECK(cudaError_t err, const char* msg = "") {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err)
                  << " - " << msg << std::endl;
        std::exit(1);
    }
}

size_t prod_shape(const std::vector<size_t>& s);

