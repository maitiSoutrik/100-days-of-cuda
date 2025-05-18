#ifndef MSE_CUH
#define MSE_CUH

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h> // For cuBLAS (though will be replaced by Thrust for mse_gpu)
#include <device_launch_parameters.h> // For blockIdx, threadIdx etc.

// Thrust includes
#include <thrust/device_ptr.h>
#include <thrust/transform.h>        // For thrust::transform
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>       // For thrust::plus, thrust::minus

// Macro for checking CUDA errors
#define CHECK_CUDA_ERROR(val) checkCuda((val), #val, __FILE__, __LINE__)
template <typename T>
void checkCuda(T result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), cudaGetErrorName(result), func);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

// Macro for checking cuBLAS errors
#define CHECK_CUBLAS_ERROR(val) checkCublas((val), #val, __FILE__, __LINE__)
inline void checkCublas(cublasStatus_t result, char const *const func, const char *const file, int const line) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error at %s:%d code=%d \"%s\" \n",
                file, line, static_cast<unsigned int>(result), func);
        // You might want to map cublasStatus_t to a string name if available/needed
        // For now, just printing the code.
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}


/**
 * @brief Calculates Mean Squared Error on the CPU.
 *
 * @param predictions Pointer to the predictions vector.
 * @param targets Pointer to the targets vector.
 * @param N Number of elements in the vectors.
 * @return The Mean Squared Error.
 */
float mse_cpu(const float* predictions, const float* targets, int N);

/**
 * @brief Calculates Mean Squared Error on the GPU.
 *
 * This function will handle memory allocation on the device, data transfers,
 * kernel launch, and result retrieval.
 *
 * @param h_predictions Pointer to the host predictions vector.
 * @param h_targets Pointer to the host targets vector.
 * @param N Number of elements in the vectors.
 * @param mse_result Pointer to store the calculated MSE.
 */
void mse_gpu(const float* h_predictions, const float* h_targets, int N, float* mse_result);

// Forward declaration for the kernel if its implementation is in mse.cu
// and not directly called from outside mse.cu's mse_gpu function.
// If mse_gpu is the sole entry point, this might not be strictly needed here.

#endif // MSE_CUH
