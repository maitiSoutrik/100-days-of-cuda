# Day 74: 2D Rotary Positional Embeddings (RoPE-2D)

## Overview

This project implements 2D Rotary Positional Embeddings (RoPE-2D) in CUDA. RoPE is a method for encoding positional information in Transformer models. While 1D RoPE (implemented in Day 73) encodes positions along a single sequence, 2D RoPE extends this concept to encode positions on a 2D grid, such as pixels in an image or tokens in a 2D layout.

The core idea is to split the embedding dimension into two halves. The first half is used to encode the row (height) position using 1D RoPE logic, and the second half is used to encode the column (width) position, also using 1D RoPE logic. Each of these halves is further divided into pairs of features, which are rotated based on their respective coordinate (height or width) and frequency.

## Implementation Details

### `rope_2d.cuh`
- Defines the `CHECK_CUDA_ERROR` macro for robust error handling.
- Declares the main function `apply_rope_2d_embeddings_gpu` which orchestrates the RoPE-2D application on the GPU.
- The function takes the device pointer to embeddings, grid dimensions (height, width), the embedding dimension, and an optional `theta_base`.
- **Constraint**: The `embedding_dim` MUST be a multiple of 4. This ensures that `embedding_dim / 2` (the dimension for each 1D RoPE application for height and width) is an even number, allowing features to be paired up correctly for rotation.

### `rope_2d.cu`
- **`rope_2d_kernel`**:
    - This CUDA kernel applies the 2D RoPE transformations.
    - Each thread in the kernel is responsible for processing a single pair of features within a token's embedding vector.
    - The kernel calculates the token's 2D coordinates (`h`, `w`) from its global 1D index.
    - It determines if the current feature pair belongs to the first half (for height encoding) or the second half (for width encoding) of the `embedding_dim`.
    - For the first half, the position `m = h` is used. For the second half, `m = w` is used.
    - The rotation angle `m * theta_k` is calculated, where `theta_k = theta_base ^ (-2k / (embedding_dim / 2))`. `k` here is the index of the feature pair *within its respective half*.
    - The feature pair `(x_0, x_1)` is then rotated:
        - `x_0' = x_0 * cos(m*theta_k) - x_1 * sin(m*theta_k)`
        - `x_1' = x_0 * sin(m*theta_k) + x_1 * cos(m*theta_k)`
- **`apply_rope_2d_embeddings_gpu`**:
    - Validates that `embedding_dim` is a multiple of 4 and that `height` and `width` are positive.
    - Calculates the total number of feature pairs to process across all tokens.
    - Configures and launches the `rope_2d_kernel`.
    - Includes error checking after kernel launch.

### `rope_2d_main.cu`
- Contains the `main` function for demonstrating the 2D RoPE implementation.
- Initializes sample 2D data (e.g., a 4x4 grid of tokens, each with an 8-dimensional embedding).
- Copies data to the GPU, calls `apply_rope_2d_embeddings_gpu`.
- Copies results back to the CPU and prints a sample of original and transformed embeddings for verification.
- Includes a basic check for NaN/Inf values in the output.

### `rope_2d_test.cu`
- Contains Google Tests for verifying the correctness of the `rope_2d_kernel`.
- **`AssertFloatVecEq`**: A helper function to compare vectors of floats with a tolerance.
- **`rope_cpu_single_pair`**: A CPU helper to apply RoPE to a single feature pair.
- **`apply_rope_2d_cpu`**: A CPU implementation of the 2D RoPE logic used as a reference to validate GPU results.
- **`BasicRotation` Test**: Uses a small grid (2x2) and a minimal valid embedding dimension (4) with simple input values (1.0, 0.0 for pairs) to check fundamental rotation logic.
- **`LargerDimensions` Test**: Uses a larger grid (8x8) and a larger embedding dimension (16) with more varied input data to test scalability and correctness under more complex conditions.
- Compares GPU output against the CPU reference implementation.

## Key CUDA Features Used
- CUDA Kernels (`__global__`) for parallel computation.
- `blockIdx`, `blockDim`, `threadIdx` for thread indexing.
- `cudaMalloc`, `cudaMemcpy`, `cudaFree` for device memory management.
- `cudaGetLastError` and `cudaGetErrorString` for error handling (via `CHECK_CUDA_ERROR`).
- Basic math functions (`cosf`, `sinf`, `powf`) within the kernel.

## Performance Considerations
- The current kernel assigns one thread per feature pair. For very large embedding dimensions or many tokens, this approach should parallelize well.
- Memory access for `embeddings` is coalesced as threads within a warp will likely access contiguous memory locations for feature pairs of the same token or adjacent tokens (depending on `embedding_dim` and `blockDim.x`).
- The calculation of `theta_k`, `cos_m_theta`, and `sin_m_theta` is done per thread. For very deep embeddings, these values could potentially be pre-calculated and stored in shared memory if multiple threads within a block were to process different tokens but the same feature pair index, but the current 1-thread-per-pair model is simpler.
- The `theta_base` and `embedding_dim` are uniform across threads, suitable for direct use or constant memory if further optimization were needed (though likely unnecessary here).

## Building and Running

### Prerequisites
- CUDA Toolkit (>= 10.0, tested with 11.x, 12.x)
- CMake (>= 3.10)
- A C++ compiler compatible with CUDA (e.g., g++)
- Google Test (will be fetched by CMake if `FetchContent` is used in the root `CMakeLists.txt`, or must be available on the system).

### Build Steps (from the root `100-days-of-cuda` directory)
1.  Ensure `day074` is added to the root `CMakeLists.txt`:
    ```cmake
    # ... other days ...
    add_subdirectory(day074)
    ```
2.  Create a build directory (if it doesn't exist) and navigate into it:
    ```bash
    mkdir -p build
    cd build
    ```
3.  Run CMake and build:
    ```bash
    cmake ..
    make -j$(nproc) # Or simply 'make'
    ```
    On Windows with MSVC:
    ```bash
    cmake ..
    cmake --build . --config Release # Or Debug
    ```

### Running the Benchmark/Demonstration
The executable will be located in `build/day074/`.
```bash
./build/day074/rope_2d_benchmark
```

### Running Tests
Tests are run via CTest from the build directory:
```bash
cd build # (if not already there)
ctest --output-on-failure -R day074_rope_2d # Run tests specifically for day074
# Or to run all tests
# ctest --output-on-failure
```

## Execution Results

Benchmark Output (`./build/day074/rope_2d_benchmark`):
```
Original Embeddings (Sample):
Original for token at (0, 0) [index 0]:
0.5488 0.5928 0.7152 0.8443 0.6028 0.8579 0.5449 0.8473 
Original for token at (1, 1) [index 5]:
0.9786 0.4736 0.7992 0.8009 0.4615 0.5205 0.7805 0.6789 

Applying 2D RoPE...

Transformed Embeddings (Sample):
Transformed for token at (0, 0) [index 0]:
0.5488 0.5928 0.7152 0.8443 0.6028 0.8579 0.5449 0.8473 
Transformed for token at (1, 1) [index 5]:
0.1302 1.0794 0.7911 0.8089 -0.1886 0.6695 0.7737 0.6867 

Verification: No NaN or Inf values found in transformed embeddings.

2D RoPE demonstration finished.
```

Test Output (`./build/day074/rope_2d_test_exec` or via CTest):
```
[==========] Running 3 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 3 tests from RoPE2DTest
[ RUN      ] RoPE2DTest.BasicRotation
[       OK ] RoPE2DTest.BasicRotation (86 ms)
[ RUN      ] RoPE2DTest.LargerDimensions
[       OK ] RoPE2DTest.LargerDimensions (1 ms)
[ RUN      ] RoPE2DTest.ZeroHeightWidth
[       OK ] RoPE2DTest.ZeroHeightWidth (0 ms)
[----------] 3 tests from RoPE2DTest (88 ms total)

[----------] Global test environment tear-down
[==========] 3 tests from 1 test suite ran. (88 ms total)
[  PASSED  ] 3 tests.
```

## Learnings and Observations

*(To be filled in after implementation and testing)*
- The primary challenge in 2D RoPE is correctly managing the indexing for height-based and width-based rotations within the single embedding vector.
- Ensuring `embedding_dim` is a multiple of 4 is crucial for the logic to correctly split dimensions for height and width, and then further into pairs.
- The CPU reference implementation is invaluable for verifying the GPU kernel's correctness, especially with floating-point operations.
- The concept extends naturally from 1D RoPE by applying it independently to two "virtual" sub-embeddings, one for each spatial dimension.

## Future Improvements
- Explore performance with extremely large grids or embedding dimensions.
- Investigate if pre-calculating `theta_k` values and storing them in constant or shared memory offers benefits for specific hardware or problem sizes.
- Extend to 3D RoPE for volumetric data.
