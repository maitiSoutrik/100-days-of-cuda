// day086/common_utils.h
#ifndef COMMON_UTILS_H
#define COMMON_UTILS_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib> // For exit, EXIT_FAILURE

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(err), cudaGetErrorName(err), func);
        // cudaDeviceReset(); // Optional: consider project's error handling strategy
        exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
inline void checkLast(const char* const file, const int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"cudaGetLastError\"\n",
                file, line, static_cast<unsigned int>(err), cudaGetErrorName(err));
        // cudaDeviceReset(); // Optional: consider project's error handling strategy
        exit(EXIT_FAILURE);
    }
}

#endif // COMMON_UTILS_H
