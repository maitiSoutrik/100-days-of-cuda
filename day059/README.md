# Day 059: Basic Ray Tracing with CUDA

## Overview
This project implements a very basic ray tracer in CUDA C++. It renders a single sphere with diffuse shading and a directional light source. The output is saved as a PPM (Portable Pixmap) image file. This example demonstrates fundamental ray tracing concepts on the GPU, including ray generation, ray-object intersection, and simple shading.

## Implementation Details

### 1. `Vec3` Structure
A simple 3D vector structure (`Vec3`) is defined in `ray_tracer.cuh` with `__device__` methods for common vector operations:
- Addition (`operator+`)
- Subtraction (`operator-`)
- Scalar Multiplication (`operator*`)
- Dot Product (`dot`)
- Normalization (`normalize`)

### 2. Constants
Global constants for image dimensions (`WIDTH`, `HEIGHT`) and sphere properties (`SPHERE_RADIUS`, `SPHERE_CENTER_X/Y/Z`) are defined in `ray_tracer.cuh`.

### 3. `render` Kernel (`ray_tracer.cu`)
The core logic resides in the `__global__ void render(unsigned char *image)` kernel:
- **Thread Mapping**: Each thread is mapped to a pixel in the output image.
- **Ray Generation**:
    - Pixel coordinates are converted to normalized screen space coordinates (`u`, `v`), including aspect ratio correction and centering the ray within each pixel. The Y-coordinate is flipped to match typical image coordinate systems.
    - The ray origin is fixed at the camera position (0,0,0).
    - The ray direction is calculated from the camera towards the (u,v) point on the view plane (assumed at z=-1.0) and then normalized.
- **Ray-Sphere Intersection**:
    - The intersection is calculated by solving a quadratic equation derived from the sphere's implicit equation `(P-C) . (P-C) = R^2` and the ray equation `P = O + tD`.
    - `oc = ray_origin - sphere_center`
    - `a = ray_dir.dot(ray_dir)` (should be 1 if `ray_dir` is normalized)
    - `b = 2.0f * oc.dot(ray_dir)`
    - `c = oc.dot(oc) - SPHERE_RADIUS * SPHERE_RADIUS`
    - `discriminant = b*b - 4*a*c`
- **Shading**:
    - If `discriminant >= 0`, an intersection exists. The smallest positive intersection time `t` is chosen.
    - The hit point is calculated: `hit_point = ray_origin + ray_dir * t`.
    - The surface normal at the hit point is `(hit_point - sphere_center).normalize()`.
    - A simple diffuse (Lambertian) shading model is applied: `intensity = fmaxf(0.0f, normal.dot(light_dir))`.
    - The light direction is a fixed normalized vector.
    - The sphere is colored with a reddish tone mixed with a slight ambient component.
- **Background**: If no intersection occurs, or the intersection is behind the camera, a solid dark blue background color is applied.

### 4. Host Code (`main.cu`)
- **Memory Management**: Allocates memory on the host (`h_image`) and device (`d_image`) for the image.
- **Kernel Launch**: Configures thread block and grid dimensions and launches the `render` kernel.
    - `threadsPerBlock(16, 16)`
    - `numBlocks((WIDTH + 15) / 16, (HEIGHT + 15) / 16)`
- **Error Checking**: Uses a `CHECK_CUDA_ERROR` macro for CUDA API calls and `cudaGetLastError()` after kernel launch. `cudaDeviceSynchronize()` ensures kernel completion.
- **Image Saving**:
    - The `save_image` function copies the rendered image from device to host.
    - It then writes the image data to `output/output.ppm` in the P6 PPM format (binary RGB).
    - An `ensure_output_directory` helper function creates the `output/` directory if it doesn't exist.

### 5. Testing (`ray_tracer_test.cu`)
Unit tests using Google Test cover:
- `Vec3` operations (addition, subtraction, scalar multiplication, dot product, normalization) by launching small test kernels.
- Basic functionality of the `render` kernel: ensures it executes without CUDA errors and that the first pixel of the output image matches the expected background color (a simple heuristic).

## Key CUDA Features Used
- **CUDA Kernels (`__global__`)**: For parallel execution of the rendering logic per pixel.
- **Device Functions (`__device__`)**: For `Vec3` operations callable from the kernel.
- **Thread Indexing (`blockIdx`, `blockDim`, `threadIdx`)**: To map threads to pixels.
- **CUDA Memory Management**: `cudaMalloc`, `cudaMemcpy`, `cudaFree`.
- **Error Handling**: `cudaGetErrorString`, `cudaGetLastError`, `cudaDeviceSynchronize`.
- **Math Functions**: `sqrtf`, `fmaxf` from `cmath` (available in CUDA device code).

## Performance Considerations
- **Parallelism**: The primary advantage is rendering all pixels in parallel.
- **Memory Access**: The `image` array is written to by many threads. This is generally fine as each thread writes to a unique location.
- **Computational Cost**: Ray-sphere intersection involves several floating-point operations per ray. More complex scenes or shading models would increase this.
- **Optimization**: This is a basic implementation. Optimizations could include:
    - Using shared memory for tile-based rendering (though less relevant for this simple forward tracer).
    - More advanced acceleration structures (e.g., BVH) for scenes with many objects (not applicable here).
    - More sophisticated sampling and anti-aliasing techniques.

## Building and Running
The project is built using CMake. Ensure CUDA Toolkit (>=10.0, compatible with sm_53) and Google Test are available in your build environment (e.g., on the Jetson Nano or CI runner).

1.  **Configure CMake from the root project directory (if not already done for other days):**
    ```bash
    cd /path/to/100-days-of-cuda
    mkdir build
    cd build
    cmake .. 
    ```
2.  **Build the project (from the `build` directory):**
    ```bash
    cmake --build . --target ray_tracer_main -j$(nproc)
    # To build tests as well
    cmake --build . --target ray_tracer_test -j$(nproc)
    ```
    Alternatively, build everything:
    ```bash
    cmake --build . -j$(nproc)
    ```
3.  **Run the executable (from the `build/day059` directory):**
    ```bash
    ./ray_tracer_main 
    ```
    The output image will be saved as `build/day059/output/output.ppm`.

4.  **Run tests (from the `build` directory):**
    ```bash
    cd /path/to/100-days-of-cuda/build
    ctest --output-on-failure -R day059_ray_tracing # Run tests for day059
    # Or directly:
    # ./day059/ray_tracer_test
    ```

## Execution Results
(Filled in on 2025-05-07 after running on Jetson Nano)

**Console Output:**

*Main Program Execution:*
```
drboom@JetNano ~/g/1/build> ./day059/ray_tracer_main 
Launching render kernel with 3072 blocks and 256 threads per block...
Image saved as output/output.ppm
Day 059: Ray Tracing with CUDA completed.
```

*Test Execution:*
```
drboom@JetNano ~/g/1/build> ./day059/ray_tracer_test 
Running main() from /home/drboom/git_repos/100-days-of-cuda/build/_deps/googletest-src/googletest/src/gtest_main.cc
[==========] Running 7 tests from 2 test suites.
[----------] Global test environment set-up.
[----------] 6 tests from Vec3Test
[ RUN      ] Vec3Test.Addition
[       OK ] Vec3Test.Addition (89 ms)
[ RUN      ] Vec3Test.Subtraction
[       OK ] Vec3Test.Subtraction (1 ms)
[ RUN      ] Vec3Test.ScalarMultiplication
[       OK ] Vec3Test.ScalarMultiplication (1 ms)
[ RUN      ] Vec3Test.DotProduct
[       OK ] Vec3Test.DotProduct (1 ms)
[ RUN      ] Vec3Test.Normalization
[       OK ] Vec3Test.Normalization (1 ms)
[ RUN      ] Vec3Test.NormalizationZeroVector
[       OK ] Vec3Test.NormalizationZeroVector (1 ms)
[----------] 6 tests from Vec3Test (95 ms total)

[----------] 1 test from RenderTest
[ RUN      ] RenderTest.KernelExecutesAndProducesOutput
[       OK ] RenderTest.KernelExecutesAndProducesOutput (22 ms)
[----------] 1 test from RenderTest (22 ms total)

[----------] Global test environment tear-down
[==========] 7 tests from 2 test suites ran. (117 ms total)
[  PASSED  ] 7 tests.
```

**Output Image (`output/output.ppm`):**
(A textual description or a link if the image is hosted elsewhere. The PPM file itself is binary.)
The image should show a reddish sphere against a dark blue background, illuminated from the top-right-front.

![output/output.ppm](output/output.ppm) 
(This Markdown link will only work if the README is viewed in an environment where `output/output.ppm` is a relative path to an actual viewable image, or if a converter is used to make a PNG/JPG for display).

## Learnings and Observations
(To be filled in after implementation and testing)
- Basic ray tracing principles are relatively straightforward to implement in CUDA.
- Mapping threads to pixels is a natural fit for GPU parallelism.
- Careful handling of floating-point precision and edge cases (e.g., discriminant values, `t` values) is important.
- The PPM format is simple for saving raw image data.

## (Optional) Future Improvements
- Add support for multiple objects (e.g., more spheres, planes).
- Implement an acceleration structure like a Bounding Volume Hierarchy (BVH) for faster intersection tests with many objects.
- Add more sophisticated shading (e.g., specular highlights, shadows from other objects).
- Implement anti-aliasing (e.g., supersampling).
- Add support for reflections and refractions.
- Read scene descriptions from a file.
