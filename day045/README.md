# Day 45: Optical Flow Gradients (Lucas-Kanade - Gradient Step)

## Overview

This day focuses on a core component of the Lucas-Kanade optical flow algorithm: calculating image gradients. Optical flow estimates the apparent motion of objects between consecutive frames in a video sequence. The Lucas-Kanade method assumes that the flow is constant in a local neighborhood and solves a system of equations based on image gradients. This implementation calculates the spatial gradients (Ix, Iy) using a simple Sobel filter and the temporal gradient (It) by differencing two consecutive frames.

## Implementation Details

-   **Input:** Two consecutive grayscale frames captured from the default camera (index 0) using OpenCV (`cv::VideoCapture`). Frame 1 is captured at time `t`, and Frame 2 is captured immediately after at time `t+1`. The frames are resized to a fixed processing size (e.g., 640x480) and converted to grayscale before processing.
-   **Gradients:**
    -   **Temporal Gradient (It):** Calculated directly as `It(x, y) = Frame2_gray(x, y) - Frame1_gray(x, y)`. This measures the change in pixel intensity over time at each location.
    -   **Spatial Gradients (Ix, Iy):** Calculated using 3x3 Sobel filters applied to Frame 1.
        -   Ix (horizontal gradient): Uses the Sobel X kernel `[-1 0 1; -2 0 2; -1 0 1]`
        -   Iy (vertical gradient): Uses the Sobel Y kernel `[-1 -2 -1; 0 0 0; 1 2 1]`
-   **CUDA Kernel (`computeGradientsKernel`):**
    -   Each thread processes one pixel (x, y).
    -   Calculates `It` for its assigned pixel.
    -   Calculates `Ix` and `Iy` using the Sobel stencil. Border pixels are handled simply by setting their gradients to zero in this version. A more robust implementation might use padding or edge clamping.
-   **Host Code:**
    -   Opens the default camera using `cv::VideoCapture`.
    -   Captures two consecutive BGR frames.
    -   Resizes and converts frames to grayscale `cv::Mat` objects.
    -   Allocates host memory for the output gradient arrays (`h_Ix`, `h_Iy`, `h_It`).
    -   Allocates device memory for input frames and output gradients.
    -   Copies input frame data (`frame1_gray.data`, `frame2_gray.data`) from host to device.
    -   Launches the `computeGradientsKernel`.
    -   Copies the resulting gradients (Ix, Iy, It) back from device to host.
    -   Performs a basic verification by counting non-zero gradients (expecting many in a real scene).
    -   Frees allocated host memory (`h_Ix`, `h_Iy`, `h_It`). OpenCV `cv::Mat` objects handle their own memory.

## Dependencies

-   **CUDA Toolkit:** Required for compilation and runtime.
-   **CMake:** Used for building the project.
-   **OpenCV:** Required for camera capture and basic image processing (grayscale conversion, resizing). Ensure OpenCV is installed on the target Jetson Nano environment.

## Key CUDA Concepts Used

-   **CUDA Kernel (`__global__`):** Parallel computation of gradients across all pixels.
-   **Thread Indexing (`blockIdx`, `blockDim`, `threadIdx`):** Mapping threads to image pixels.
-   **Device Memory Management:** `cudaMalloc`, `cudaMemcpy`, `cudaFree`.
-   **Error Handling:** `CHECK_CUDA_ERROR` macro for robust API call checking.
-   **Stencil Computation:** Basic implementation within the kernel for Sobel filters (potential for shared memory optimization in future work).

## Performance Considerations

-   The current implementation uses global memory for all reads (input frames) and writes (output gradients) during the Sobel calculation.
-   **Potential Optimization:** Using shared memory for the Sobel stencil operation could significantly improve performance by reducing global memory accesses. Threads within a block could cooperatively load the required 3x3 neighborhood into shared memory, perform the calculations, and then write the results back to global memory.
-   Border handling is currently basic (setting gradients to zero). More sophisticated methods (clamping, mirroring) might be needed for real images but add slight complexity.
-   The use of live camera data introduces variability depending on the scene and motion.

## Building and Running

**Note:** Compilation and execution should be done on the target platform (Jetson Nano) or a compatible environment as per project rules.

1.  **Navigate to the build directory:**
    ```bash
    cd /path/to/100-days-of-cuda/build
    ```
2.  **Configure using CMake:** (Run from the build directory)
    ```bash
    cmake ..
    ```
3.  **Build the executable:** (Run from the build directory)
    ```bash
    cmake --build . --target optical_flow_gradients
    # Or build all targets (if the CI/CD script relies on this)
    # cmake --build .
    ```
4.  **Run the executable:** (Located in `build/day045/`)
    ```bash
    ./day045/optical_flow_gradients
    ```

## Execution Results

(Output from running on Jetson Nano via CI/CD)

```
Day 45: Optical Flow Gradient Calculation (CUDA with OpenCV Camera Input)
[ WARN:0@0.767] global cap_gstreamer.cpp:1728 open OpenCV | GStreamer warning: Cannot query video position: status=0, value=-1, duration=-1
Captured two consecutive frames from camera 0.
Processing Dimensions: 640 x 480
Computing gradients on GPU...
GPU computation complete.

Sample Gradient Values (Center Pixel (320, 240)):
    Ix: -23.00, Iy: 91.00, It: 8.00

Verification (counts of non-zero gradients):
  Ix > 0: 289095 / 307200 pixels
  Iy > 0: 290073 / 307200 pixels
  It > 0: 271388 / 307200 pixels
  Max absolute It: 44.00

Day 45 finished successfully.
```
*Note: The GStreamer warning is common on some platforms and usually doesn't indicate a functional problem with frame capture.*

## Learnings and Observations

-   Integrated OpenCV for real-time camera input, making the gradient calculation applicable to live video streams.
-   The process involves capturing frames, preprocessing (resizing, grayscale conversion), transferring data to the GPU, executing the kernel, and retrieving results.
-   Calculating gradients on real camera feeds shows expected non-zero values for Ix, Iy, and It, reflecting texture and motion in the scene.
-   The direct application of Sobel filters in the kernel remains straightforward but highlights the opportunity for shared memory optimization.

## Future Improvements

-   Implement the shared memory optimization for the Sobel filter calculation to improve performance.
-   Implement proper boundary handling (e.g., clamp-to-edge) for more accurate gradients near image borders.
-   Proceed to the next steps of the Lucas-Kanade algorithm (forming and solving the local 2x2 linear systems).

## References

-   Lucas–Kanade method - Wikipedia: [https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method](https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method)
-   Sobel operator - Wikipedia: [https://en.wikipedia.org/wiki/Sobel_operator](https://en.wikipedia.org/wiki/Sobel_operator)
