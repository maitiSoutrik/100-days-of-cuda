# Day 095: CUDA Barnsley Fern Fractal Generator

## Overview
This project implements a Barnsley Fern fractal generator using CUDA C++. The goal is to leverage GPU parallelism to compute the many iterations required for generating the fractal and output the result as a color PPM image, displaying a green fern on a black background.

## Implementation Details
- The fractal is generated using an Iterated Function System (IFS) with four affine transformations, chosen probabilistically.
- A CUDA kernel is responsible for generating points in parallel. Each thread computes a sequence of points.
- `cuRAND` is used for random number generation on the GPU to select transformations.
- The generated points are mapped to a 2D image buffer, where pixel hit counts are recorded.
- The host code manages memory, launches the kernel, and saves the resulting image buffer as a PPM (P6 binary) file, with hit pixels colored green and unhit pixels black.

## Dependencies
- CUDA Toolkit (version compatible with Jetson Nano, including the `cuRAND` library)
- CMake (>=3.10)
- Google Test (fetched by CMake)

## Key CUDA Features Used
- CUDA Kernels (`__global__` functions)
- `cuRAND` library for GPU-accelerated random number generation.
- Atomic operations (e.g., `atomicAdd`) for updating the image buffer safely from multiple threads.
- Device memory management (`cudaMalloc`, `cudaMemcpy`, `cudaFree`).

## Performance Considerations
- The generation of fractal points is highly parallelizable, making it suitable for GPU acceleration.
- The number of points and iterations can be scaled to produce higher-detail fractals.
- Normalization of pixel values for the PGM output is done on the CPU after copying the data back from the device.

## Building and Running
1.  Ensure you have the CUDA Toolkit (including the `cuRAND` library) installed and correctly configured on your Jetson Nano build environment.
2.  Navigate to the `day095` directory.
3.  Create a build directory: `mkdir build && cd build`
4.  Run CMake: `cmake ..`
5.  Compile: `make`
6.  Run the executable: `./barnsley_fern_app`
    This will generate a `barnsley_fern.ppm` file in the build directory.
7.  Run the tests (optional but recommended): `ctest` or `./barnsley_fern_test` (if in the same directory or path is adjusted).

## Execution Results
The program successfully generates a `barnsley_fern.ppm` file depicting a green fern on a black background.

Console output from running the application and tests on the Jetson Nano:
```
drboom@JetNano ~/g/1/build> ./day095/barnsley_fern_app 
Day 095: Barnsley Fern Fractal Generator
Image dimensions: 1000x1000
Total points to generate: 200000000
Threads per block: 256
Number of blocks: 781
Points per thread: 1001
cuRAND states initialized.
Launching Barnsley Fern generation kernel...
Kernel execution finished.
Copying image buffer from device to host...
Copy finished.
Saved fern to barnsley_fern.ppm
drboom@JetNano ~/g/1/build> ./day095/barnsley_fern_test 
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from BarnsleyFernTest
[ RUN      ] BarnsleyFernTest.KernelExecutionSmokeTest
[       OK ] BarnsleyFernTest.KernelExecutionSmokeTest (104 ms)
[----------] 1 test from BarnsleyFernTest (104 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (104 ms total)
[  PASSED  ] 1 test.
```

*Image of the PPM output (converted to PNG/JPG for display in Markdown):*
(User to add `barnsley_fern.ppm` converted to a web-friendly format here, e.g., using `convert barnsley_fern.ppm barnsley_fern.png`)

## Learnings and Observations
- Initial attempts to save as a grayscale PGM image encountered issues where the image appeared black due to normalization problems with hit counts.
- Debugging involved simplifying the PGM saving logic to a binary black/white representation, which confirmed the point generation was working.
- To introduce color (green fern), the output format was changed from PGM to PPM (P6 binary). This required modifying the saving function to write RGB triplets for each pixel.
- Type mismatches in `std::min`/`std::max` (double vs. float literals) caused a build error on the target, resolved by using float literals (`0.0f`, `255.0f`).

## Future Improvements
- Direct PNG output using a library like `stb_image_write.h`.
- Interactive parameter adjustments (e.g., number of points, colors).
- Exploration of different fractal types.
- More sophisticated color mapping based on point density or iteration count.
