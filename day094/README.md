# Day 094: CUDA Implementation of Forward and Simplified Reverse Diffusion Steps

## Overview

This project implements the core mechanics of Denoising Diffusion Probabilistic Models (DDPMs) using CUDA: the forward diffusion process (adding noise) and a simplified version of the reverse diffusion process (denoising). The goal is to understand the fundamental operations at a low level and see how they can be parallelized on a GPU.

This implementation does **not** include the neural network (e.g., U-Net) typically used in DDPMs to predict the noise \(\epsilon_\theta\). Instead, for the reverse step, we use a placeholder for the predicted noise to demonstrate the kernel structure.

## Implementation Details

The project consists of:
1.  `ddpm_kernels.cuh` / `ddpm_kernels.cu`: Contains the CUDA kernels for:
    *   Initializing cuRAND states for random number generation on the GPU.
    *   `forward_diffusion_step_kernel`: Applies one step of Gaussian noise to the input data according to the formula:
        \(x_t = \sqrt{1-\beta_t} \cdot x_{t-1} + \sqrt{\beta_t} \cdot \epsilon\)
        where \(\epsilon \sim \mathcal{N}(0, I)\).
    *   `simplified_reverse_diffusion_step_kernel`: Implements a simplified reverse step:
        \(x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_{\text{predicted}} \right) + \sigma_t z\)
        where \(\alpha_t = 1 - \beta_t\), \(z \sim \mathcal{N}(0, I)\) (if \(\sigma_t > 0\)), and \(\epsilon_{\text{predicted}}\) is a placeholder. In this demo, if `d_epsilon_predicted` is `nullptr`, the kernel uses `x_t[idx] * 0.1f` as a very rough placeholder for `epsilon_predicted[idx]`.

2.  `ddpm_main.cu`: A demonstration program that:
    *   Initializes a sample dataset (a sine wave).
    *   Simulates a few steps of the forward diffusion process, printing statistics at each step.
    *   Simulates a few steps of the simplified reverse diffusion process using the output of the forward pass, printing statistics.

3.  `ddpm_test.cu`: Google Tests for the CUDA kernels, checking:
    *   Basic properties of the forward diffusion (data changes, values are finite, approximate mean/stddev).
    *   Basic properties of the simplified reverse diffusion (data changes, values are finite, weak check on mean/stddev change).

## Key CUDA Features Used

*   **CUDA Kernels:** For parallel execution of diffusion steps.
*   **cuRAND:** For generating Gaussian random numbers (`epsilon` and `z`) directly on the GPU using `curand_normal()`. `curandState` is initialized per thread.
*   **Device Memory Management:** `cudaMalloc`, `cudaMemcpy`, `cudaFree`.
*   **Error Handling:** `CHECK_CUDA_ERROR` macro for robust error checking.
*   **Thread Indexing:** Standard `blockIdx.x * blockDim.x + threadIdx.x` for 1D data.

## Performance Considerations

*   **Parallelism:** Each element (e.g., pixel) in the data can be processed independently in both forward and reverse steps, making these operations highly parallelizable on the GPU.
*   **Random Number Generation:** Using cuRAND on the device avoids costly CPU-GPU transfers of random numbers. Each thread maintains its own cuRAND state.
*   **Memory Access:** Kernels perform element-wise operations, leading to coalesced memory access if data is laid out linearly.
*   **Simplified Reverse Step:** The current reverse step is simplified. A full DDPM involves a neural network inference to predict \(\epsilon\), which would be the most computationally intensive part. This demo focuses only on applying the diffusion equations.

## Building and Running

The project uses CMake. Ensure you have the CUDA Toolkit installed.

**Build (from the `100-days-of-cuda/build` directory, assuming you are there):**
```bash
cd /path/to/100-days-of-cuda/build # Or your build directory
cmake .. 
make day094_ddpm_steps_ddpm_demo day094_ddpm_steps_ddpm_tests 
# Or simply 'make' if you want to build everything
```
This will create executables `ddpm_demo` and `ddpm_tests` in the `build/day094/` directory.

**Running the Demo:**
```bash
./day094/ddpm_demo
```

**Running Tests:**
```bash
./day094/ddpm_tests
```
Or using CTest from the build directory:
```bash
ctest --output-on-failure -R day094_ddpm_steps_ddpm_tests
```

## Execution Results

The `ddpm_demo` executable produces output showing the statistics of the data as it undergoes forward and then simplified reverse diffusion steps.

**Output from `./day094/ddpm_demo`:**
```
Initial Data (h_x0): [0.0000, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009...]
Initial Data (h_x0) Stats: Mean = 0.0000, StdDev = 0.7071

--- cuRAND States Initialized ---

--- Forward Diffusion Simulation ---
Forward Step t=1 (beta_t=0.00012):
  Data (h_x_t): [0.0086, -0.0030, -0.0109, 0.0163, -0.0012, 0.0004, 0.0165, 0.0060, -0.0073, 0.0173...]
  Data (h_x_t) Stats: Mean = -0.0000, StdDev = 0.7071
Forward Step t=2 (beta_t=0.00014):
  Data (h_x_t): [-0.0128, 0.0037, -0.0060, 0.0096, -0.0059, -0.0058, 0.0212, 0.0119, -0.0416, 0.0232...]
  Data (h_x_t) Stats: Mean = 0.0000, StdDev = 0.7072
Forward Step t=3 (beta_t=0.00016):
  Data (h_x_t): [-0.0109, 0.0003, -0.0148, 0.0069, -0.0010, 0.0093, 0.0412, 0.0126, -0.0631, 0.0448...]
  Data (h_x_t) Stats: Mean = 0.0000, StdDev = 0.7072
Forward Step t=4 (beta_t=0.00018):
  Data (h_x_t): [-0.0177, -0.0148, -0.0153, -0.0059, 0.0126, -0.0079, 0.0590, 0.0139, -0.0460, 0.0461...]
  Data (h_x_t) Stats: Mean = 0.0000, StdDev = 0.7073
Forward Step t=5 (beta_t=0.00020):
  Data (h_x_t): [-0.0216, -0.0138, -0.0005, -0.0159, -0.0002, -0.0234, 0.0706, 0.0125, -0.0427, 0.0501...]
  Data (h_x_t) Stats: Mean = 0.0000, StdDev = 0.7074

--- Simplified Reverse Diffusion Simulation ---
Reverse Step, target t=4 (from t=5):
  Data (h_x_reverse): [-0.0284, -0.0047, 0.0047, -0.0025, -0.0032, -0.0381, 0.0469, 0.0006, -0.0725, 0.0394...]
  Data (h_x_reverse) Stats: Mean = 0.0000, StdDev = 0.7071
Reverse Step, target t=3 (from t=4):
  Data (h_x_reverse): [-0.0517, -0.0208, 0.0285, -0.0036, -0.0091, -0.0494, 0.0397, 0.0081, -0.0930, 0.0379...]
  Data (h_x_reverse) Stats: Mean = -0.0000, StdDev = 0.7068
Reverse Step, target t=2 (from t=3):
  Data (h_x_reverse): [-0.0465, -0.0159, 0.0621, 0.0282, 0.0109, -0.0473, 0.0475, -0.0130, -0.1050, 0.0408...]
  Data (h_x_reverse) Stats: Mean = -0.0000, StdDev = 0.7064
Reverse Step, target t=1 (from t=2):
  Data (h_x_reverse): [-0.0328, 0.0005, 0.0683, 0.0283, -0.0356, -0.0434, 0.0242, -0.0300, -0.1143, 0.0233...]
  Data (h_x_reverse) Stats: Mean = 0.0000, StdDev = 0.7059
Reverse Step, target t=0 (from t=1):
  Data (h_x_reverse): [-0.0442, -0.0152, 0.0766, 0.0256, -0.0317, -0.0381, 0.0232, -0.0269, -0.1124, 0.0295...]
  Data (h_x_reverse) Stats: Mean = 0.0000, StdDev = 0.7053

--- DDPM Demo Finished ---
```

**Output from `./day094/ddpm_tests`:**
```
[==========] Running 2 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 2 tests from DDPMKernelsTest
[ RUN      ] DDPMKernelsTest.ForwardDiffusionStep
[       OK ] DDPMKernelsTest.ForwardDiffusionStep (187 ms)
[ RUN      ] DDPMKernelsTest.SimplifiedReverseDiffusionStep
[       OK ] DDPMKernelsTest.SimplifiedReverseDiffusionStep (100 ms)
[----------] 2 tests from DDPMKernelsTest (288 ms total)

[----------] Global test environment tear-down
[==========] 2 tests from 1 test suite ran. (288 ms total)
[  PASSED  ] 2 tests.
```
The forward diffusion steps show the standard deviation of the data slightly increasing as noise is added, as expected. The mean remains close to zero. The simplified reverse diffusion steps show the standard deviation gradually decreasing, indicating some noise removal, though not a perfect reconstruction due to the placeholder \(\epsilon_{\text{predicted}}\). The tests pass, confirming the basic functionality of the kernels.

## Learnings and Observations

*   The core diffusion equations are relatively simple to implement as CUDA kernels.
*   Managing cuRAND states is crucial for generating noise on the GPU.
*   The forward process predictably increases the entropy/randomness of the data.
*   The simplified reverse process demonstrates the mathematical structure, but its effectiveness heavily depends on the (missing) \(\epsilon_{\text{predicted}}\) term. A real DDPM's power comes from the learned noise predictor.
*   This exercise highlights the separation of the fixed diffusion process mathematics from the learned denoising model.

## Future Improvements

*   Implement the full DDPM sampling loop (T steps starting from pure noise).
*   Integrate a pre-trained noise prediction model (e.g., a small ONNX model run via TensorRT, or a very simple hardcoded "predictor").
*   Precompute \(\beta_t, \alpha_t, \bar{\alpha}_t\) schedules.
*   Implement different variance schedules (e.g., cosine).
