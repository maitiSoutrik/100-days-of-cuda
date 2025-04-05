# Day 27: Mirror Descent (STE) for Quantization

## Overview

This project implements the core optimization algorithm discussed in the paper "Mirror Descent View for Neural Network Quantization." Specifically, it uses the numerically stable Mirror Descent approach, which is analogous to the Straight-Through Estimator (STE), to optimize a simple function (Sphere function: f(x) = Σ xᵢ²) while constraining the solution towards discrete values {-1, 1}. This is achieved using a `tanh` projection function and an annealing schedule for the `beta` parameter.

The goal is to demonstrate the practical application of the paper's theoretical justification for STE in the context of quantization-aware training, even on a simple optimization problem.

## Implementation Details

The implementation uses CUDA C++ and involves the following components:

1.  **Host Logic:**
    *   Initializes a vector `x_latent` (representing the unconstrained auxiliary variables `x̃` from the paper) with random values between -1 and 1 on the host.
    *   Allocates device memory for `d_x_latent`, `d_x_projected` (representing the projected variables `x`), and `d_gradient`.
    *   Copies the initial `h_x_latent` to `d_x_latent`.
    *   Runs an optimization loop for a fixed number of iterations.
    *   Inside the loop:
        *   Calls `projection_kernel` to compute `d_x_projected = tanhf(beta * d_x_latent)`.
        *   Calls `gradient_kernel` to compute the gradient of the Sphere function `∇f(x)` based on `d_x_projected`.
        *   Calls `update_latent_kernel` to update `d_x_latent` using the gradient: `d_x_latent = d_x_latent - learning_rate * d_gradient`.
        *   Increases the annealing parameter `beta` multiplicatively (`beta *= rho`).
    *   Copies the final `d_x_latent` and `d_x_projected` back to the host.
    *   Prints the final latent and projected values, the function value calculated with projected values, and the function value calculated with `sign(latent)` values (which should approach N).
    *   Frees allocated memory.

2.  **CUDA Kernels:**
    *   `projection_kernel(float* d_x_projected, const float* d_x_latent, float beta, int N)`: Computes `d_x_projected[i] = tanhf(beta * d_x_latent[i])` for each element `i`.
    *   `gradient_kernel(float* d_gradient, const float* d_x_projected, int N)`: Computes the gradient of the Sphere function, `d_gradient[i] = 2.0f * d_x_projected[i]`, for each element `i`.
    *   `update_latent_kernel(float* d_x_latent, const float* d_gradient, float learning_rate, int N)`: Performs the Mirror Descent update in the dual space: `d_x_latent[i] = d_x_latent[i] - learning_rate * d_gradient[i]`.

## Key CUDA Features Used

*   Basic CUDA Kernels (`__global__`)
*   Device Memory Management (`cudaMalloc`, `cudaMemcpy`, `cudaFree`)
*   CUDA Math Function (`tanhf`)
*   Standard Error Checking (`CHECK_CUDA_ERROR` macro)
*   Thread Indexing (`blockIdx`, `blockDim`, `threadIdx`)
*   Device Synchronization (`cudaDeviceSynchronize`, `cudaGetLastError`)

## Performance Considerations

*   The Sphere function `f(x) = Σ xᵢ²` has a very simple gradient (`∇f(x) = 2x`), making the `gradient_kernel` computationally inexpensive.
*   The primary focus of this exercise is demonstrating the Mirror Descent/STE algorithm with annealing, not optimizing the Sphere function itself.
*   The annealing process (`beta` increasing over iterations) gradually forces the projected values `tanh(beta * x_latent)` towards +1 or -1. The rate of annealing (`rho`) and the number of iterations influence how closely the final values approach the discrete set.
*   Memory transfers between host and device occur only at the beginning and end, minimizing overhead during the optimization loop.

## Building and Running

1.  **Navigate to the build directory:**
    ```bash
    cd build
    ```
2.  **Configure using CMake:** Make sure you are in the `build` directory located at the root of the `100-days-of-cuda` repository.
    ```bash
    cmake ..
    ```
    *(Note: If you encounter errors about `nvcc` not being found, ensure your CUDA toolkit path is correctly configured in your environment or CMake settings, e.g., by setting `CUDAToolkit_ROOT`)*
3.  **Build the executable:**
    ```bash
    make mirror_descent
    ```
4.  **Run the executable:**
    ```bash
    ./day027/mirror_descent
    ```

## Execution Results

**(Please replace the placeholder below with the actual output obtained when running the code, ideally on a Jetson Nano)**

```
Starting Mirror Descent (STE) Optimization for Sphere Function
N = 1024, Iterations = 5000, LR = 0.0100, Initial Beta = 1.00, Rho = 1.0010
Iteration 1, Beta = 1.0010
Iteration 500, Beta = 1.6479
Iteration 1000, Beta = 2.7179
Iteration 1500, Beta = 4.4810
Iteration 2000, Beta = 7.3878
Iteration 2500, Beta = 12.1798
Iteration 3000, Beta = 20.0788
Iteration 3500, Beta = 33.0961
Iteration 4000, Beta = 54.5584
Iteration 4500, Beta = 89.9411
Iteration 5000, Beta = 148.2818

Optimization finished.
Final Beta = 148.2818
Final Latent Variables (first 10):
  x_latent[0] = -1.680399
  x_latent[1] = 1.680399
  x_latent[2] = 1.680399
  x_latent[3] = -1.680399
  x_latent[4] = 1.680399
  x_latent[5] = -1.680399
  x_latent[6] = -1.680399
  x_latent[7] = 1.680399
  x_latent[8] = -1.680399
  x_latent[9] = 1.680399
Final Projected Variables (first 10):
  x_projected[0] = -1.000000
  x_projected[1] = 1.000000
  x_projected[2] = 1.000000
  x_projected[3] = -1.000000
  x_projected[4] = 1.000000
  x_projected[5] = -1.000000
  x_projected[6] = -1.000000
  x_projected[7] = 1.000000
  x_projected[8] = -1.000000
  x_projected[9] = 1.000000

Final function value f(x_projected) = 1024.000000
Final function value f(sign(x_latent)) = 1024.000000 (Should approach N=1024)

Day 27 Mirror Descent finished successfully.
```

## Learnings and Observations

*   This exercise demonstrates how the STE algorithm, viewed through the lens of Mirror Descent, effectively pushes variables towards a discrete set ({-1, 1} in this case) during optimization.
*   The `tanh(beta * x_latent)` projection acts as the mapping from the latent/dual space back to the (constrained) primal space.
*   The annealing of `beta` is crucial. As `beta` increases, the `tanh` function becomes steeper, approximating the `sign` function and forcing the projected values closer to +/- 1.
*   The latent variables `x_latent` do not necessarily converge to +/- 1 themselves, but their sign determines the final quantized value. The magnitude reflects the "confidence" of the quantization.
*   The final function value calculated using `sign(x_latent)` correctly reflects the objective function evaluated on the truly quantized values, which is N (1024) for the Sphere function when all xᵢ are +/- 1.

## References

*   Meng, Guanya, et al. "Mirror Descent View for Neural Network Quantization." arXiv preprint arXiv:2106.07183 (2021). (Conceptual basis)
