#ifndef ELU_ACTIVATION_CUH
#define ELU_ACTIVATION_CUH

#include <cuda_runtime.h>
#include <iostream> // For std::cerr, std::endl
#include <cmath>    // For expf

// CUDA Error Checking Macro
#ifndef CHECK_CUDA_ERROR
#define CHECK_CUDA_ERROR(val) checkCuda((val), #val, __FILE__, __LINE__)
inline void checkCuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(result)
                  << " \"" << cudaGetErrorString(result) << "\" func=" << func << std::endl;
        // cudaDeviceReset(); // Optional: Resets the CUDA state, might be too drastic for some apps
        // exit(99); // Optional: Exits the program, consider throwing an exception instead
    }
}
#endif // CHECK_CUDA_ERROR

/**
 * @brief Computes the Exponential Linear Unit (ELU) activation on the GPU.
 *
 * For each element x in the input array, the ELU function is defined as:
 *   f(x) = x, if x > 0
 *   f(x) = alpha * (exp(x) - 1), if x <= 0
 *
 * @param d_input Pointer to the input array on the device.
 * @param d_output Pointer to the output array on the device.
 * @param n Number of elements in the arrays.
 * @param alpha The alpha parameter for the ELU function (typically 1.0f).
 */
void elu_activation_kernel_wrapper(float* d_input, float* d_output, int n, float alpha);

/**
 * @brief Host function to manage ELU activation on the GPU.
 *
 * Allocates memory on the device, copies input data from host to device,
 * launches the ELU kernel, copies results from device to host, and frees device memory.
 *
 * @param h_input Pointer to the input array on the host.
 * @param h_output Pointer to the output array on the host (results will be stored here).
 * @param n Number of elements in the arrays.
 * @param alpha The alpha parameter for the ELU function (typically 1.0f).
 */
void elu_activation_gpu(float* h_input, float* h_output, int n, float alpha);

/**
 * @brief CPU implementation of the ELU activation function for verification.
 *
 * @param input Pointer to the input array.
 * @param output Pointer to the output array.
 * @param n Number of elements in the arrays.
 * @param alpha The alpha parameter for the ELU function.
 */
void elu_activation_cpu(const float* input, float* output, int n, float alpha);

#endif // ELU_ACTIVATION_CUH
