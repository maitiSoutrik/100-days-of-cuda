#ifndef DDPM_KERNELS_CUH
#define DDPM_KERNELS_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h> // For GPU-side random number generation
#include <cstdio> // For printf in kernels (debugging)

// CUDA Error Checking Macro
#define CHECK_CUDA_ERROR(val) checkCuda((val), #val, __FILE__, __LINE__)
inline void checkCuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), cudaGetErrorName(result), func);
        cudaDeviceReset();
        exit(99);
    }
}

// --- Forward Diffusion Step ---
// Kernel to apply one step of Gaussian noise to an image (or batch of images/data)
// x_t = sqrt(1 - beta_t) * x_{t-1} + sqrt(beta_t) * epsilon
__global__ void forward_diffusion_step_kernel(
    float* x_t_minus_1,      // Input data (e.g., image pixels) at step t-1
    float* x_t,              // Output data at step t
    float beta_t,            // Noise variance for this step
    curandState* states,     // cuRAND states for generating epsilon
    int N                    // Total number of elements (e.g., pixels)
);

// Wrapper function to launch the forward diffusion kernel
void launch_forward_diffusion_step(
    float* d_x_t_minus_1,
    float* d_x_t,
    float beta_t,
    curandState* d_states,
    int N,
    dim3 threads_per_block,
    dim3 num_blocks
);


// --- Simplified Reverse Diffusion Step ---
// Kernel for a simplified reverse step.
// For this example, let's assume epsilon_theta (predicted noise) is given or very simple.
// x_{t-1} = (1/sqrt(alpha_t)) * (x_t - ( (1-alpha_t) / sqrt(1-alpha_bar_t) ) * epsilon_predicted) + sigma_t * z
// where alpha_t = 1 - beta_t
// For simplicity in this initial implementation, we might directly subtract a portion of x_t as if it's noise,
// or use a very simple epsilon_predicted.
// Let's make epsilon_predicted an input for now.
__global__ void simplified_reverse_diffusion_step_kernel(
    float* x_t,                  // Input data at step t (noisy)
    float* x_t_minus_1,          // Output data at step t-1 (less noisy)
    float* epsilon_predicted,    // Predicted noise by a hypothetical model (or simple placeholder)
    float alpha_t,               // 1.0f - beta_t
    float one_minus_alpha_bar_t_sqrt, // sqrt(1 - alpha_bar_t)
    float sigma_t,               // Noise for stochasticity in reverse step (can be 0 for deterministic)
    curandState* states,         // cuRAND states for generating z (if sigma_t > 0)
    int N                        // Total number of elements
);

// Wrapper function to launch the simplified reverse diffusion kernel
void launch_simplified_reverse_diffusion_step(
    float* d_x_t,
    float* d_x_t_minus_1,
    float* d_epsilon_predicted,
    float alpha_t,
    float one_minus_alpha_bar_t_sqrt,
    float sigma_t,
    curandState* d_states,
    int N,
    dim3 threads_per_block,
    dim3 num_blocks
);

// --- cuRAND State Initialization ---
__global__ void setup_curand_states_kernel(curandState *state, unsigned long seed, int N);

void launch_setup_curand_states(
    curandState *d_states, 
    unsigned long seed, 
    int N,
    dim3 threads_per_block,
    dim3 num_blocks
);

#endif // DDPM_KERNELS_CUH
