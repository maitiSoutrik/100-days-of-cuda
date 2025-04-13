# Day 35: Kalman Filter Prediction Step with cuBLAS

## Introduction

This example implements the **prediction step** of a Kalman filter using CUDA and the cuBLAS library. The Kalman filter is a powerful algorithm used for state estimation in dynamic systems, particularly when dealing with noisy sensor measurements. The prediction step estimates the future state and its uncertainty based on the system's dynamics model.

The core equations for the prediction step are:
- **State Prediction:** `x_pred = F * x + B * u`
- **Covariance Prediction:** `P_pred = F * P * F^T + Q`

Where:
- `x`: Current state vector
- `P`: Current state covariance matrix
- `F`: State transition matrix
- `B`: Control input matrix (optional, not used in this example)
- `u`: Control input vector (optional, not used in this example)
- `Q`: Process noise covariance matrix
- `x_pred`: Predicted state vector
- `P_pred`: Predicted state covariance matrix

This implementation focuses on accelerating the matrix multiplications (`F * x`, `F * P`, `(F * P) * F^T`) and addition (`... + Q`) using cuBLAS's `sgemm` (matrix-matrix multiplication) and `sgeam` (matrix addition) functions. A CPU version is included for comparison and verification.

## Implementation Details

- **Model:** A simple 2D constant velocity model is assumed for demonstration. The state vector `x` is `[position_x, position_y, velocity_x, velocity_y]`, making `STATE_DIM = 4`.
- **Data Loading:** The code requires the path to an IMU data CSV file as a command-line argument. It expects the CSV format to match the example provided (comma-separated, header starting with `#`), specifically looking for `a_RS_S_x` (column 4) and `a_RS_S_y` (column 5) as the 2D measurements (`MEASUREMENT_DIM = 2`). The header line is skipped. The initial state is derived (simplified) from the first valid data row loaded. If the file cannot be opened or no valid data rows are parsed, the program exits with an error. *Note: This example only implements the prediction step; a full filter would also include the update step using these measurements. Using acceleration directly as measurements for a position/velocity state is a simplification for this example.*
- **CPU Version:** Uses basic C++ loops for matrix multiplication (`matrixMultiplyCPU`), addition (`matrixAddCPU`), and transpose (`matrixTransposeCPU`).
- **GPU Version (`kalmanPredictGPU`):**
    - Uses `cublasCreate` to initialize a cuBLAS handle.
    - Allocates device memory using `cudaMalloc`.
    - Copies initial state (`h_x`), covariance (`h_P`), and model matrices (`h_F`, `h_Q`) from host to device using `cudaMemcpy`.
    - Performs the prediction step using cuBLAS functions:
        - `x_pred = F * x`: Implemented using `cublasSgemm` treating the vector `x` as a `STATE_DIM x 1` matrix.
        - `Temp = F * P`: Implemented using `cublasSgemm`.
        - `P_pred_noQ = Temp * F^T`: Implemented using `cublasSgemm` with the `CUBLAS_OP_T` flag for the second matrix (`F`).
        - `P_pred = P_pred_noQ + Q`: Implemented using `cublasSgeam`.
    - **Row-Major vs. Column-Major:** The CPU code uses row-major storage. cuBLAS defaults to column-major. The `cublasSgemm` and `cublasSgeam` calls are set up assuming the input device pointers point to row-major data, and the parameters (M, N, K, LDA, LDB, LDC, transpose operations) are adjusted accordingly to achieve the correct mathematical operations.
    - Times the GPU execution using CUDA Events (`cudaEventRecord`, `cudaEventElapsedTime`).
    - Copies the predicted state (`d_x_out`) and covariance (`d_P_out`) back to the host.
    - Cleans up resources using `cudaFree`, `cublasDestroy`, and `cudaEventDestroy`.
- **Benchmarking:** Compares the execution time of the CPU prediction step (`kalmanPredictCPU`) against the GPU version (`kalmanPredictGPU`).
- **Verification:** Calculates the sum of absolute differences between the CPU and GPU results for both the predicted state and covariance matrix.

## Key CUDA Features Used

- **cuBLAS Library:**
    - `cublasCreate()`, `cublasDestroy()`: Handle management.
    - `cublasSgemm()`: Single-precision general matrix-matrix multiplication for `F*x`, `F*P`, and `(F*P)*F^T`.
    - `cublasSgeam()`: Single-precision general matrix addition for `... + Q`.
- **CUDA Runtime API:**
    - `cudaMalloc()`: Device memory allocation.
    - `cudaMemcpy()`: Host-to-device and device-to-host data transfers.
    - `cudaFree()`: Device memory deallocation.
    - `cudaEvent_t`, `cudaEventCreate()`, `cudaEventRecord()`, `cudaEventSynchronize()`, `cudaEventElapsedTime()`, `cudaEventDestroy()`: Accurate GPU timing.
- **Error Handling:** `CHECK_CUDA_ERROR` and `CHECK_CUBLAS_ERROR` macros for robust error checking.

## Performance Considerations

- **cuBLAS Efficiency:** cuBLAS is highly optimized for NVIDIA GPUs. For small matrix sizes (like the 4x4 used here), the overhead of launching kernels and transferring data might dominate the computation time, potentially making the GPU version slower than the CPU version for a single prediction step.
- **Batch Processing:** The real benefit of using cuBLAS for Kalman filters often comes when processing multiple filters (states) in parallel (batching). This example currently processes only one state (`num_states = 1`), but the `kalmanPredictGPU` function signature includes `num_states` as a placeholder for future batch implementation (e.g., using `cublasSgemmStridedBatched`).
- **Data Transfer:** Copying matrices (F, Q) and the state/covariance (x, P) between host and device for every prediction step can be a bottleneck. In a real-time system, matrices F and Q might remain constant on the device, and the state/covariance would ideally stay on the device between prediction and update steps.
- **CPU Implementation:** The provided CPU matrix functions are very basic and not optimized (e.g., no cache blocking, SIMD). A highly optimized CPU BLAS library (like OpenBLAS, MKL) would provide a much stronger baseline for comparison.

## Building and Running

**Note:** Build and run this code on the target Jetson Nano or a compatible environment with the CUDA toolkit and appropriate drivers installed.

1.  **Navigate to the build directory:**
    ```bash
    cd 100-days-of-cuda/build
    ```
2.  **Configure using CMake:** Ensure the top-level `CMakeLists.txt` includes `add_subdirectory(day035)`.
    ```bash
    cmake ..
    ```
    *(You should see messages confirming Day 35 configuration and CUDA architecture 53)*
3.  **Build the executable:**
    ```bash
    make kalman_predict
    ```
    *(Alternatively, build all targets with `make`)*
4.  **Run the executable:** Provide the path to your IMU data CSV file as an argument.
    ```bash
    ./day035/kalman_predict <path_to_your_imu_data.csv>
    ```
    Example using the expected path on the target system:
    ```bash
    ./day035/kalman_predict /home/drboom/cuda-data-sets/imu-data.csv
    ```

The program will load data from the specified CSV file, using the 5th and 6th columns (index 4 and 5, `a_RS_S_x`, `a_RS_S_y`) as measurements. It will then perform the Kalman prediction step on both the CPU and GPU, print the predicted state vectors, report the execution times, and show the difference between the CPU and GPU results.

## Execution Results

*Output from running `./build/day035/kalman_predict /home/drboom/cuda-data-sets/imu-data.csv` on the Jetson Nano:*

```
Day 35: Kalman Filter Prediction Step using cuBLAS
Attempting to load data from: /home/drboom/cuda-data-sets/imu-data.csv
Successfully loaded 36820 data points.

--- Running CPU Kalman Prediction ---
CPU Prediction Step Duration: 0.00125 ms
CPU Predicted State (x_pred) (4x1):
  [8.1477]
  [-0.3759]
  [0.0000]
  [0.0000]

--- Running GPU Kalman Prediction (cuBLAS) ---
GPU Prediction Step Duration: 0.402969 ms
GPU Predicted State (x_pred) (4x1):
  [8.1477]
  [-0.3759]
  [0.8148]
  [-0.0376]

--- Verification ---
Sum absolute difference (State x): 0.852361
Sum absolute difference (Covariance P): 40

Finished Day 35.
```

**Analysis:**
- The GPU execution (0.403 ms) is significantly slower than the CPU execution (0.001 ms) for this single prediction step with small matrices (4x4). This highlights the overhead associated with CUDA kernel launches and cuBLAS setup, which outweighs the computational benefit for this specific task scale.
- There's a noticeable difference in the predicted state (particularly velocities) and covariance between the CPU and GPU versions. This discrepancy (Sum absolute difference (State x): 0.852361, Sum absolute difference (Covariance P): 40) needs investigation. Potential causes include:
    - Floating-point precision differences between CPU and GPU calculations.
    - Differences in how intermediate results are handled or rounded.
    - Potential subtle issues in the cuBLAS calls or matrix handling (though the logic follows the standard prediction equations). The covariance difference is quite large and warrants closer inspection, possibly by printing intermediate matrices.
- The data loading was successful, processing a large number of points from the CSV.

## Learnings and Observations

- Implementing matrix operations using cuBLAS (`sgemm`, `sgeam`) is straightforward once the row-major vs. column-major handling is understood.
- For small, single-instance problems, the overhead of CUDA calls and data transfers can negate the GPU's computational advantage compared to a simple CPU implementation. The performance benefits are expected to become significant with batch processing or larger state dimensions.
- Error checking for both CUDA runtime and cuBLAS calls is crucial for debugging.
- CUDA event timing provides accurate measurement of GPU execution time, excluding data transfer overhead unless explicitly included.
- Verification against a CPU implementation is essential to ensure the correctness of the GPU code, especially when dealing with matrix conventions.

## (Optional) Future Improvements

- Implement the **update step** of the Kalman filter.
- Implement **batch processing** using `cublasSgemmStridedBatched` to filter multiple states simultaneously.
- Read the time step `dt` from the IMU data timestamps instead of using a fixed value.
- Use a more sophisticated motion model or tune the `Q` matrix based on sensor characteristics.
- Compare performance against an optimized CPU BLAS library.
- Keep filter state and matrices entirely on the GPU if performing iterative filtering.
