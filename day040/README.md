# Day 40: Texture Memory for Image Interpolation (Bilinear)

## Overview

This project explores the use of CUDA's texture memory for optimizing image processing tasks, specifically focusing on bilinear interpolation for image upscaling. Texture memory provides hardware-accelerated filtering and boundary handling mechanisms through specialized cache, making it efficient for operations involving non-integer coordinates or spatial locality.

We implement a simple 2x image upscaling process:
1.  Load a grayscale image using OpenCV.
2.  Convert the image to 32-bit float format.
3.  Bind the image data to a `cudaArray` and create a `cudaTextureObject`.
4.  Configure the texture object for bilinear filtering (`cudaFilterModeLinear`) and edge clamping (`cudaAddressModeClamp`).
5.  Launch a CUDA kernel where each thread corresponds to a pixel in the output (upscaled) image.
6.  Each thread calculates its corresponding *normalized* coordinates in the source texture.
7.  The thread uses `tex2D<float>()` to fetch the bilinearly interpolated value from the texture at the calculated coordinates.
8.  The interpolated value is written to the output device buffer.
9.  The resulting upscaled image is copied back to the host, converted to 8-bit, and saved using OpenCV.
10. A CPU-based bilinear interpolation using `cv::resize` is performed for comparison.

## Implementation Details

*   **Input/Output:** Reads a grayscale PNG image (defaults to `../day014/lena_gray.png`, can be overridden via command line). Writes the GPU-interpolated and CPU-interpolated output images to a `./output_interpolated/` directory. Accepts an optional upscale factor (default 2.0) and output filename via command line arguments.
*   **Data Types:** Uses `float` (CV_32FC1) for image data during processing to leverage texture filtering precision. The final output is converted back to `uchar` (CV_8UC1).
*   **CUDA Array:** The input image data is copied to a `cudaArray`, which is optimized for texture fetching. `cudaMemcpy2DToArray` is used for this transfer.
*   **Texture Object:**
    *   `cudaResourceDesc`: Describes the `cudaArray` as the resource.
    *   `cudaTextureDesc`: Configures the texture behavior:
        *   `addressMode = cudaAddressModeClamp`: Coordinates outside [0, 1] are clamped to the edge pixel values.
        *   `filterMode = cudaFilterModeLinear`: Enables hardware bilinear filtering between texels.
        *   `readMode = cudaReadModeElementType`: Reads the data as its native type (`float`).
        *   `normalizedCoords = 1`: The kernel uses normalized coordinates (0.0 to 1.0) for fetching.
    *   `cudaCreateTextureObject`: Creates the texture object handle.
*   **Kernel (`bilinear_interpolation_kernel`):**
    *   Launched with a 2D grid matching the output image dimensions.
    *   Calculates the normalized texture coordinates `(u, v)` corresponding to the *center* of the current output pixel `(x, y)`. This ensures correct alignment: `u = (x + 0.5f) / outWidth`, `v = (y + 0.5f) / outHeight`.
    *   `tex2D<float>(texObj, u, v)`: Fetches the interpolated value using the hardware texture unit.
    *   Writes the result to the global memory output buffer `d_output`.
*   **CPU Comparison:** Uses `cv::resize(..., cv::INTER_LINEAR)` for a reference implementation.

## Key CUDA Features Used

*   **Texture Memory:** The core concept demonstrated.
    *   `cudaArray`: Optimized memory layout for texture fetching.
    *   `cudaChannelFormatDesc`: Describes the data format (`float`).
    *   `cudaMallocArray`, `cudaFreeArray`: Allocation/deallocation.
    *   `cudaMemcpy2DToArray`: Transferring data to the array.
    *   `cudaResourceDesc`, `cudaTextureDesc`: Configuration.
    *   `cudaCreateTextureObject`, `cudaDestroyTextureObject`: Lifetime management.
    *   `cudaTextureObject_t`: Texture object handle passed to the kernel.
    *   `tex2D<float>()`: Intrinsic function for 2D texture fetching with filtering.
*   **Kernel Launch:** Standard 2D grid-stride loop pattern adapted for image processing.
*   **Error Handling:** `CHECK_CUDA_ERROR` macro.

## Performance Considerations

*   **Texture Cache:** Texture fetches benefit from a dedicated cache hierarchy, potentially offering higher bandwidth and lower latency compared to global memory reads, especially when access patterns exhibit spatial locality (nearby threads access nearby texels).
*   **Hardware Filtering:** The bilinear interpolation calculation is performed by specialized hardware units, offloading the computation from the CUDA cores.
*   **Address Calculation & Boundary Handling:** Texture hardware also handles coordinate normalization, boundary mode application (clamping), and address calculations efficiently.
*   **Overhead:** Creating and destroying texture objects has some overhead, making them more suitable when the texture will be accessed multiple times or by many threads.
*   **Data Type:** Texture interpolation typically works best with native types like `float`.

In this specific case of simple 2x upscaling, the performance difference between GPU texture-based interpolation and CPU `cv::resize` might vary. For more complex interpolation schemes, larger images, or repeated sampling, the benefits of texture memory become more pronounced.

## Building and Running

**Prerequisites (on Jetson Nano or build environment):**

*   CUDA Toolkit (>= 10.2)
*   CMake (>= 3.10)
*   GCC/G++
*   OpenCV (`libopencv-dev` with `core`, `imgproc`, `imgcodecs` modules)

**Build Steps (from the root `100-days-of-cuda` directory):**

1.  Ensure the `day040` subdirectory has been added to the root `CMakeLists.txt`.
2.  Create/navigate to the build directory: `mkdir -p build && cd build`
3.  Configure using CMake: `cmake ..`
4.  Build the specific target: `make texture_interpolation` (or `make` for all)

**Running:**

1.  Navigate to the executable location: `cd build/day040`
2.  Run with defaults (Lena 2x upscale): `./texture_interpolation`
3.  Run with a different image: `./texture_interpolation path/to/your/image.png`
4.  Run with a different image and upscale factor: `./texture_interpolation path/to/your/image.png 3.0`
5.  Run specifying image, factor, and output file: `./texture_interpolation ../day014/lena_gray.png 4.0 ./output_interpolated/lena_4x_gpu.png`

The program will print timing information and save the GPU and CPU interpolated images in the `./output_interpolated/` directory relative to where the executable is run.

## Execution Results / Output (Jetson Nano CI)

The following output was captured during the CI run on the Jetson Nano:

**1. File Input Mode (Default):**
```
Running texture_interpolation with default file input...
Output preview (default) (see /home/drboom/git_repos/100-days-of-cuda/logs/day040_texture_interpolation.log for full output):
Mode: File Input
Input image: ../day014/lena_gray.png
Upscale factor: 2.00
Output image GPU (last frame for camera): ./output_interpolated/lena_gray_interpolated_gpu_x2.0.png
Output image CPU (last frame for camera): ./output_interpolated/lena_gray_interpolated_cpu_x2.0.png
Created output directory: ./output_interpolated
Loaded input image: 512 x 512 channels: 3
Input dimensions: 512 x 512
Output dimensions: 1024 x 1024
Starting processing for 1 frame(s)...
Frame: 0, Timestamp_ms: 1745023306003, GPU Interpolation Time: 9.598 ms
Processing last frame, copying result and saving...
Saved GPU interpolated image to: ./output_interpolated/lena_gray_interpolated_gpu_x2.0.png
Saved CPU interpolated image to: ./output_interpolated/lena_gray_interpolated_cpu_x2.0.png
CPU Interpolation Time (last frame): 2.385 ms
Processing complete.
Cleaning up resources...
Cleanup complete.
```
*(Note: The input image `lena_gray.png` was loaded as 3 channels and converted in the C++ code; the processing uses the grayscale version. CPU time is significantly faster here, likely due to OpenCV optimizations for this specific task on ARM.)*

**2. Camera Input Mode (`--camera 0`):**
```
Running texture_interpolation with camera input (index 0)...
Output preview (camera) (see /home/drboom/git_repos/100-days-of-cuda/logs/day040_texture_interpolation_camera.log for full output):
Frame: 88, Timestamp_ms: 1745023315559, GPU Interpolation Time: 28.875 ms
Frame: 89, Timestamp_ms: 1745023315599, GPU Interpolation Time: 29.726 ms
Frame: 90, Timestamp_ms: 1745023315640, GPU Interpolation Time: 31.119 ms
Frame: 91, Timestamp_ms: 1745023315679, GPU Interpolation Time: 28.911 ms
Frame: 92, Timestamp_ms: 1745023315719, GPU Interpolation Time: 30.193 ms
Frame: 93, Timestamp_ms: 1745023315759, GPU Interpolation Time: 29.384 ms
Frame: 94, Timestamp_ms: 1745023315800, GPU Interpolation Time: 31.044 ms
Frame: 95, Timestamp_ms: 1745023315839, GPU Interpolation Time: 28.730 ms
Frame: 96, Timestamp_ms: 1745023315878, GPU Interpolation Time: 29.544 ms
Frame: 97, Timestamp_ms: 1745023315919, GPU Interpolation Time: 30.157 ms
Frame: 98, Timestamp_ms: 1745023315959, GPU Interpolation Time: 30.043 ms
Frame: 99, Timestamp_ms: 1745023315999, GPU Interpolation Time: 29.792 ms
Processing last frame, copying result and saving...
Saved GPU interpolated image to: ./output_interpolated/camera_frame_interpolated_gpu_x2.0.png
Saved CPU interpolated image to: ./output_interpolated/camera_frame_interpolated_cpu_x2.0.png
CPU Interpolation Time (last frame): 2.773 ms
Processing complete.
Cleaning up resources...
Camera released.
Cleanup complete.
```
*(Note: Frame times include camera capture, H->D copy, grayscale conversion kernel, texture creation, interpolation kernel, D->H copy (last frame), and saving (last frame). CPU comparison time is only for the `cv::resize` operation on the last frame.)*

*(Check the `build/day040/output_interpolated` directory on the Jetson for the generated PNG files.)*

## Learnings and Observations

*   Texture memory provides a convenient high-level abstraction for filtered memory access, simplifying interpolation kernels.
*   Understanding normalized coordinates and how they map to pixel centers is crucial for correct interpolation.
*   The setup involves creating `cudaArray`, `cudaResourceDesc`, `cudaTextureDesc`, and `cudaTextureObject`, which adds some boilerplate compared to direct global memory access.
*   Texture memory performance gains are most evident with spatial locality and when hardware filtering/boundary modes are beneficial. For simple linear access, global memory might be competitive.
*   Using `float` textures is common for preserving precision during interpolation.

## Future Improvements

*   Adapt to use the camera input similar to Day 30, performing interpolation on live frames.
*   Implement different interpolation modes (e.g., nearest neighbor by changing `filterMode`).
*   Handle color images (e.g., using `float4` textures or separate textures per channel).
*   Compare performance against a manual bilinear interpolation kernel using global memory.
