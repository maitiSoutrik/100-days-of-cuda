#pragma once
#include <cstddef> // For size_t

template <typename T, size_t BLOCK_SIZE>
__global__ void gemm_upper_tri(const T* A, const T* B, T* C, int n);
