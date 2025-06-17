# Day 098: Advanced Dynamic Parallelism - Mandelbrot Set with Adaptive Mesh Refinement

## Overview

This project implements a Mandelbrot set fractal generator using CUDA, featuring Adaptive Mesh Refinement (AMR) through dynamic parallelism. The core idea is to dynamically launch more detailed computations (finer-grained kernel launches) in regions of the fractal that exhibit high complexity, such as the intricate boundaries. This approach aims to optimize rendering by focusing computational effort where it's most needed, rather than uniformly computing the entire image at the highest resolution.

This builds upon concepts from Day 047 (Dynamic Parallelism Introduction) and Day 095 (Barnsley Fern Fractal, another fractal example).

## Implementation Details

The implementation consists of several key components:

1.  **`Region` Structure**: A `struct` defining a rectangular area in the complex plane to be computed. It includes:
    *   `x_min, x_max, y_min, y_max`: Complex plane coordinates.
    *   `width_px, height_px`: Dimensions of the region in pixels.
    *   `start_x_global, start_y_global`: Offset of this region within the global image.
    *   `current_depth`: Current recursion depth for AMR.

2.  **`mandelbrot_iterations` (Device Function)**:
    *   Calculates the number of iterations for a given complex point `(cx, cy)` to escape the Mandelbrot set condition (`|z|^2 <= 4`).
    *   Returns the iteration count, capped by `max_iterations`.

3.  **`mandelbrot_kernel` (Global Kernel)**:
    *   Takes a `Region` as input and computes the Mandelbrot iteration count for each pixel within that region.
    *   Maps pixel coordinates to complex numbers.
    *   Calls `mandelbrot_iterations` for each point.
    *   Writes the resulting color (based on iteration count) to the correct position in the global `image_data` buffer.
    *   **Dynamic Parallelism Aspect (Simplified in current host-driven version)**: In a full device-driven dynamic parallelism model, this kernel (or a designated thread/block within it) would analyze the computed sub-region. If the sub-region requires further refinement (e.g., high variance in iteration counts) and `current_depth` is less than `max_depth`, it would dynamically launch new instances of `mandelbrot_kernel` for sub-divided regions. The current version uses a host-side loop to simulate this queue and decision process for simplicity, but the `CMakeLists.txt` is set up to enable device-side dynamic parallelism (`-rdc=true` and linking `cudadevrt`).

4.  **`check_refinement_kernel` (Global Kernel - Placeholder/Conceptual)**:
    *   This kernel is designed to be called after `mandelbrot_kernel` computes a region.
    *   It would analyze the pixel data of the just-computed region (e.g., calculate color variance).
    *   If the variance exceeds `refinement_threshold`, it would set a flag indicating that this region needs to be subdivided.
    *   The current host-driven `generate_mandelbrot_amr` function performs a simplified version of this logic on the host by always subdividing if `current_depth < max_depth`.

5.  **`generate_mandelbrot_amr` (Host Function)**:
    *   Manages the overall AMR process.
    *   Initializes a queue of `Region` structures, starting with the entire image.
    *   Iteratively processes regions from the queue:
        *   If `current_depth >= max_depth`, it launches `mandelbrot_kernel` to compute the region directly without further subdivision.
        *   If `current_depth < max_depth`:
            1.  Launches `mandelbrot_kernel` to compute the current region.
            2.  (Simplified Host Logic) Subdivides the current region into four smaller sub-regions.
            3.  Adds these new sub-regions to the processing queue.
    *   This host-side management simulates the recursive subdivision. A true device-side dynamic parallelism approach would have kernels enqueueing new work items (regions) to a device-side queue or launching child kernels directly.
    *   Copies the final image data from device to host.

6.  **Main Application (`mandelbrot_main.cu`)**:
    *   Sets up image dimensions, Mandelbrot parameters (coordinate range, max iterations), and AMR parameters (max depth, refinement threshold).
    *   Calls `generate_mandelbrot_amr`.
    *   Saves the resulting image as a PGM (Portable GrayMap) file.
    *   Includes basic timing for the generation process.

7.  **Testing (`mandelbrot_amr_test.cu`)**:
    *   Uses Google Test to verify the `mandelbrot_iterations` device function for points known to be inside and outside the set.
    *   A helper kernel `test_mandelbrot_iterations_kernel` is used to call the `__device__` function from the host via a kernel launch.

## Key CUDA Features Used

*   **Dynamic Parallelism (`cudaLaunchDevice`)**: Although the current main control loop is host-driven for simplicity, the project is structured and compiled (`-rdc=true`, linking `cudadevrt`) to support device-side kernel launches. The `mandelbrot_kernel` is intended to be capable of launching child kernels for refined regions. The `check_refinement_kernel` would typically be launched by a parent kernel to decide on further subdivision.
*   **Kernel Logic for Subdivision**: The host function `generate_mandelbrot_amr` demonstrates the logic for subdividing a `Region` into four smaller sub-regions. This logic would be mirrored in a device-side dynamic launch.
*   **Managing Recursion Depth**: The `Region` struct includes `current_depth`, and the `max_depth` parameter controls how many levels of refinement are performed.
*   **Standard Kernel Launch Syntax**: `<<<grid_dim, block_dim>>>`
*   **Device Memory Management**: `cudaMalloc`, `cudaMemset`, `cudaMemcpy`, `cudaFree`.
*   **Error Handling**: `CHECK_CUDA_ERROR` macro, `cudaGetLastError`, `cudaDeviceSynchronize`.
*   **`__device__` functions**: For `mandelbrot_iterations`.
*   **`__global__` functions**: For `mandelbrot_kernel` and `check_refinement_kernel`.

## Performance Considerations

*   **Overhead of Dynamic Launches**: Dynamic parallelism introduces some overhead for launching kernels from the device. This is most beneficial when the work launched is substantial enough to offset this overhead.
*   **Load Balancing**: AMR can improve load balancing by focusing work on complex areas. However, very small refined regions might lead to underutilization if not managed carefully.
*   **Refinement Strategy**: The effectiveness of AMR heavily depends on the heuristic used to decide when and where to refine. A simple variance check is a common starting point. The `refinement_threshold` parameter tunes this.
*   **Recursion Depth Management**: `max_depth` is crucial. Too deep can lead to excessive kernel launches and overhead. Too shallow might not provide enough detail.
*   **Memory Access**: Kernels access global memory for the image buffer. Coalesced access is generally good for image processing tasks like this.
*   **Host vs. Device Driven AMR**: The current implementation uses a host-driven queue for managing regions. A fully device-driven approach (kernels launching child kernels and managing a device-side queue) would avoid host-device synchronization within the AMR loop, potentially improving performance for deep recursion, but adds complexity.

*(Performance analysis results from Jetson Nano will be added here after execution.)*

## Building and Running

**Prerequisites:**
*   CUDA Toolkit (>= 10.0, compatible with sm_53)
*   CMake (>= 3.18)
*   A C++ compiler supporting C++14 (e.g., g++)
*   Google Test (will be fetched by CMake if not found)

**Build Steps (on Jetson Nano or compatible cross-compilation environment):**

1.  Ensure you are in the root directory of the `100-days-of-cuda` project.
2.  Update the root `CMakeLists.txt` to include this day:
    ```cmake
    # ... other days ...
    add_subdirectory(day098)
    ```
3.  Create a build directory and navigate into it:
    ```bash
    mkdir build
    cd build
    ```
4.  Run CMake and build:
    ```bash
    cmake ..
    make -j$(nproc) 
    # or specifically for this day:
    # make mandelbrot_main mandelbrot_amr_test 
    ```
    *Note: The `-rdc=true` flag (relocatable device code) and linking `cudadevrt` are essential for dynamic parallelism and are set in `day098/CMakeLists.txt`.*

**Running the Application:**
After successful compilation, the executable will be in the `build/day098/` directory.
```bash
./build/day098/mandelbrot_main
```
This will generate an image named `mandelbrot_amr_output.pgm` in the directory where you run the executable (likely the `build` directory).

**Running Tests:**
```bash
./build/day098/mandelbrot_amr_test
# Or using ctest from the build directory
# ctest -R day098_mandelbrot_amr # (If tests are correctly registered with CTest via gtest_discover_tests)
```

## Execution Results

**Console output from `mandelbrot_main` on Jetson Nano:**
```
drboom@JetNano ~/g/1/build> ./day098/mandelbrot_main 
Generating Mandelbrot set with AMR...
Image size: 800x600
Max iterations: 500
Max AMR depth: 3
Refinement threshold: 0.1
Mandelbrot generation took: 946.145 ms
Image saved as mandelbrot_amr_output.pgm
Successfully generated Mandelbrot set.
```
The generated image `mandelbrot_amr_output.pgm` (469KB) was successfully created in the build directory.

**Test execution output on Jetson Nano:**
```
drboom@JetNano ~/g/1/build> ./day098/mandelbrot_amr_test
[==========] Running 4 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 4 tests from MandelbrotIterationsTest
[ RUN      ] MandelbrotIterationsTest.PointInsideSet
[       OK ] MandelbrotIterationsTest.PointInsideSet (92 ms)
[ RUN      ] MandelbrotIterationsTest.PointOutsideSet
[       OK ] MandelbrotIterationsTest.PointOutsideSet (1 ms)
[ RUN      ] MandelbrotIterationsTest.AnotherPointOutsideSet
[       OK ] MandelbrotIterationsTest.AnotherPointOutsideSet (1 ms)
[ RUN      ] MandelbrotIterationsTest.ComplexBoundaryPoint
[       OK ] MandelbrotIterationsTest.ComplexBoundaryPoint (1 ms)
[----------] 4 tests from MandelbrotIterationsTest (97 ms total)

[----------] Global test environment tear-down
[==========] 4 tests from 1 test suite ran. (97 ms total)
[  PASSED  ] 4 tests.
```
All tests passed successfully.

## Learnings and Observations

*   **Complexity of True Dynamic Parallelism**: Implementing a fully device-driven dynamic parallelism solution with robust queue management and refinement heuristics is complex. The current host-driven approach simplifies the control flow while still allowing the core computation kernels to be developed and tested.
*   **Compilation Flags**: `-rdc=true` (relocatable device code) and linking against `cudadevrt` are crucial for enabling dynamic parallelism. Without these, `cudaLaunchDevice` calls from within a kernel will fail.
*   **Refinement Heuristic is Key**: The quality and efficiency of AMR depend heavily on the logic used to decide when to refine a region. A simple variance check is a common starting point, but more sophisticated methods exist.
*   **Debugging Dynamic Parallelism**: Debugging kernels launched dynamically can be more challenging. `printf` from device code (used sparingly) and careful logging are important. NVIDIA Nsight tools can also be very helpful.
*   **Trade-offs**: AMR trades increased complexity and potential launch overhead for potentially faster rendering of detailed fractals by avoiding over-computation in smooth areas.

## Future Improvements

*   Implement a fully device-driven AMR using a device-side work queue (e.g., using atomics for queue management).
*   Develop a more sophisticated `check_refinement_kernel` based on pixel variance or edge detection within a computed region.
*   Allow configurable block/grid sizes for the dynamically launched kernels.
*   Experiment with different `max_depth` and `refinement_threshold` values to observe their impact on performance and image quality.
*   Add colorization to the Mandelbrot image instead of grayscale.
*   Integrate with an image library (like OpenCV, if available on Jetson) for saving in more common formats (PNG, JPG).

## References

*   NVIDIA CUDA C++ Programming Guide (Chapter on Dynamic Parallelism)
*   Wikipedia: Mandelbrot Set
*   Various online tutorials on Mandelbrot generation and adaptive mesh refinement.
