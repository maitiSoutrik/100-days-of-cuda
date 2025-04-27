# Day 49: Mini-Project - Accelerated Perception Component

## Overview
This project combines several image processing kernels developed in previous days into a single pipeline running on the GPU. The goal is to simulate a basic perception component that captures a frame from a camera, processes it through grayscale conversion, Gaussian blurring, Sobel edge detection, and finally counts the number of detected edge pixels using a parallel reduction kernel. This demonstrates kernel sequencing and integration with live camera input via OpenCV.

## Implementation Details
*   **Input:** Captures a single frame from the default camera (index 0) using OpenCV (`cv::VideoCapture`).
*   **Pipeline Stages:**
    1.  **Grayscale Conversion:** Converts the captured color frame (assuming BGR format from OpenCV) to single-channel grayscale using standard luminance calculation (e.g., 0.299*R + 0.587*G + 0.114*B). (Kernel adapted from concepts in Day 30).
    2.  **Gaussian Blur:** Applies a 5x5 Gaussian blur to the grayscale image to reduce noise before edge detection. (Kernel adapted from convolution concepts, e.g., Day 7).
    3.  **Sobel Edge Detection:** Computes image gradients (Gx, Gy) using Sobel operators on the blurred image, then calculates the gradient magnitude (sqrt(Gx^2 + Gy^2)). (Kernel implemented based on Sobel operators).
    4.  **Edge Pixel Counting:** Uses a parallel reduction kernel to count the number of pixels where the Sobel magnitude exceeds a predefined threshold (e.g., 100). (Kernel adapted from reduction techniques, e.g., Day 4 or Day 33).
*   **CUDA Streams:** (Optional Implementation) CUDA streams can be used to manage the asynchronous execution of kernels and memory transfers, potentially overlapping computation and data movement between stages for improved throughput.

## Key CUDA Features Used
*   CUDA C++ Kernels (`__global__`, `__device__`) for parallel computation.
*   Device Memory Management (`cudaMalloc`, `cudaMemcpy`, `cudaFree`).
*   Thread Hierarchy (Grid, Blocks, Threads).
*   OpenCV for image loading/saving (on the host).
*   Parallel Reduction Algorithm.
*   (Optional) CUDA Streams (`cudaStreamCreate`, `cudaStreamDestroy`, kernel launches with stream parameter, `cudaMemcpyAsync`).

## Performance Considerations
*   **Kernel Sequencing:** The output of one kernel serves as the input to the next, requiring careful management of device memory buffers.
*   **Memory Access:** Kernels (especially convolution/Sobel) should aim for coalesced memory access where possible. Shared memory could be used for optimization in blur/Sobel, but is not implemented in this basic version.
*   **Stream Overlap:** If using streams, overlap between kernel execution and data transfers (if any were needed between stages beyond kernel launches) could hide latency. The primary benefit here would be overlapping the execution of independent kernels if the pipeline allowed, or overlapping H2D/D2H transfers with kernel execution.
*   **Reduction Efficiency:** The efficiency of the reduction kernel impacts the final counting step.

## Building and Running
**(Note: Compilation and execution are intended for the Jetson Nano or a compatible environment with CUDA Toolkit, CMake, OpenCV, and a connected camera.)**

1.  **Navigate to the Build Directory:**
    ```bash
    cd /path/to/100-days-of-cuda/build
    ```
2.  **Configure using CMake:**
    ```bash
    cmake ..
    ```
    *(Ensure the `day049` subdirectory is added to the root `CMakeLists.txt`)*
3.  **Build the Executable:**
    ```bash
    make day049_perception_pipeline
    ```
4.  **Run the Executable:**
    Ensure a camera is connected and accessible as device 0.
    ```bash
    ./day049/day049_perception_pipeline
    ```
    *(The program will attempt to access the default camera, capture one frame, process it, and print the detected edge count to the console.)*

## Execution Results / Output
**(Actual output from running on Jetson Nano with a connected camera.)**

```
drboom@JetNano ~/g/1/build> ./day049/day049_perception_pipeline 
Day 49: Accelerated Perception Pipeline (Internal Camera Capture)
[ WARN:0@0.771] global cap_gstreamer.cpp:1728 open OpenCV | GStreamer warning: Cannot query video position: status=0, value=-1, duration=-1
Attempting to capture frame from camera...
Frame captured successfully.
Captured Frame Info: (Width: 1280, Height: 720, Channels: 3)
----------------------------------------
Perception Pipeline Completed.
Edge Threshold: 100.0
Detected Edge Pixels: 71310
----------------------------------------
Finished successfully.
```
*(Note: The GStreamer warning is related to video stream properties and did not prevent the program from capturing and processing the frame.)*

## Learnings and Observations
*   Demonstrates how to chain multiple CUDA kernels together.
*   Highlights the need for intermediate device memory buffers.
*   Shows integration with host-side libraries like OpenCV for I/O.
*   Provides a basis for exploring stream-based optimization.

## (Optional) Future Improvements
*   Implement shared memory optimizations for Gaussian blur and Sobel kernels.
*   Perform more detailed performance analysis with `nvprof` or Nsight Systems.
*   Add more sophisticated perception steps (e.g., feature detection).
*   Compare stream-based execution vs. default stream execution.
*   Save intermediate or final processed images using OpenCV.
