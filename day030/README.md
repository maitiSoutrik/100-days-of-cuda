# Day 30: Headless USB Camera Processing with CUDA (Grayscale + Average Intensity)

## Overview

This project demonstrates capturing frames from a USB camera connected to the Jetson Nano, processing them headlessly on the GPU using CUDA, and logging the results. Specifically, it captures frames, converts them to grayscale using a CUDA kernel, calculates the average intensity of each frame using a parallel reduction kernel, and prints timestamped results to the console.

This is useful for scenarios where real-time monitoring or analysis is needed without a graphical display, such as in embedded systems or remote monitoring applications.

## Implementation Details

1.  **Camera Capture (OpenCV):**
    *   Uses `cv::VideoCapture` to open and read frames from the specified USB camera (default index 0).
    *   Retrieves frame dimensions (`width`, `height`).
    *   Includes basic error checking for camera opening and frame capture.
    *   Ensures frames are in `CV_8UC3` (3-channel unsigned char) format, converting if necessary.

2.  **Memory Management:**
    *   `cv::Mat frame`: Host memory for the captured frame, managed by OpenCV.
    *   `d_rgb_input (uchar3*)`: Device memory to store the raw RGB frame transferred from the host.
    *   `d_gray_output (float*)`: Device memory to store the calculated grayscale values (as floats in the range [0, 255]).
    *   `d_sum_output (float*)`: Device memory for intermediate partial sums during the reduction.
    *   `d_final_sum (float*)`: Device memory to hold the single final sum after the reduction is complete.
    *   `h_final_sum (float)`: Host variable to receive the final sum from the device.
    *   `h_avg_intensity (float)`: Host variable to store the calculated average intensity.

3.  **CUDA Kernels:**
    *   `rgb_to_gray_kernel`: Each thread processes one pixel. It reads a `uchar3` RGB value, calculates the luminance using the standard formula (0.299\*R + 0.587\*G + 0.114\*B), and writes the result as a float to `d_gray_output`.
    *   `reduce_sum_kernel`: Implements a parallel sum reduction using shared memory, adapted from Day 4. It handles arbitrary input sizes (`N`). A two-pass approach is used:
        *   **Pass 1:** Reduces the `d_gray_output` array (size `num_pixels`) into `d_sum_output` (size `num_blocks_for_reduction`).
        *   **Pass 2:** If `num_blocks_for_reduction > 1`, a second kernel launch (with a single block) reduces `d_sum_output` into `d_final_sum` (size 1). If only one block was needed in the first pass, the result is simply copied from `d_sum_output[0]` to `d_final_sum`.

4.  **Main Loop:**
    *   Iterates for a fixed number of frames (`MAX_FRAMES = 100`).
    *   Measures the processing time for each frame using `std::chrono`.
    *   Captures a frame.
    *   Transfers the frame from Host to Device (`cudaMemcpyHostToDevice`).
    *   Launches `rgb_to_gray_kernel`.
    *   Launches `reduce_sum_kernel` (Pass 1).
    *   Launches `reduce_sum_kernel` (Pass 2, if needed) or copies the single partial sum.
    *   Transfers the final sum from Device to Host (`cudaMemcpyDeviceToHost`).
    *   Calculates the average intensity (`h_final_sum / num_pixels`).
    *   Logs the frame number, system timestamp (milliseconds), average intensity, and frame processing time to the console.

5.  **Error Handling:** Uses the `CHECK_CUDA_ERROR` macro for CUDA API calls. Includes checks for OpenCV camera and frame validity.

## Key CUDA Features Used

*   **CUDA Kernels (`__global__`):** For parallel execution of grayscale conversion and reduction.
*   **Device Memory Management:** `cudaMalloc`, `cudaMemcpy` (HostToDevice, DeviceToHost, DeviceToDevice), `cudaFree`.
*   **Thread Hierarchy:** `blockIdx`, `blockDim`, `threadIdx`.
*   **Shared Memory (`__shared__`):** Used within `reduce_sum_kernel` for efficient parallel reduction within a block.
*   **Synchronization (`__syncthreads()`):** Used within the reduction kernel.
*   **Error Handling:** `cudaGetLastError`, `cudaGetErrorString`.

## Performance Considerations

*   **Data Transfer:** The `cudaMemcpyHostToDevice` call for transferring each full frame from the CPU to the GPU is likely the most significant bottleneck. For higher performance, techniques like mapped memory or direct V4L2 buffer access with CUDA interop could be explored, but are more complex.
*   **Kernel Efficiency:** The grayscale kernel is memory-bound (limited by global memory bandwidth). The reduction kernel is generally efficient due to shared memory usage.
*   **Synchronization:** `cudaMemcpy` calls involve implicit synchronization. `cudaGetLastError` is used for explicit error checking after kernel launches.
*   **Headless Operation:** Running without a display avoids rendering overhead.

## Building and Running

**Prerequisites (on Jetson Nano or build environment):**

*   CUDA Toolkit (>= 10.2 recommended for Jetson Nano)
*   CMake (>= 3.10)
*   GCC/G++ compiler compatible with CUDA
*   OpenCV (`libopencv-dev`): Install using `sudo apt-get update && sudo apt-get install libopencv-dev`. Ensure it has V4L2/GStreamer support if using standard builds.
*   A USB camera connected and recognized by the system (e.g., check with `ls /dev/video*`).

**Build Steps (from the root `100-days-of-cuda` directory):**

1.  Ensure the `day030` subdirectory has been added to the root `CMakeLists.txt`.
2.  Create a build directory: `mkdir build && cd build`
3.  Configure using CMake: `cmake ..`
4.  Build the specific target: `make day030_camera_intensity` (or just `make` to build all)

**Running:**

1.  Navigate to the executable location: `cd build/day030`
2.  Run the executable: `./camera_avg_intensity`
3.  (Optional) Specify a different camera index: `./camera_avg_intensity 1`

The program will print log messages to the console for each processed frame.

## Execution Results / Output

The program will output lines similar to the following for each frame processed:

```
Using camera index: 0
Camera opened successfully. Frame dimensions: 640 x 480 (307200 pixels)
Starting frame capture and processing for 100 frames...
Frame: 0, Timestamp_ms: 1744208515123, Avg Intensity: 115.3421, Frame Time: 15.83 ms
Frame: 1, Timestamp_ms: 1744208515145, Avg Intensity: 115.8876, Frame Time: 14.99 ms
Frame: 2, Timestamp_ms: 1744208515168, Avg Intensity: 116.1010, Frame Time: 15.20 ms
...
Frame: 98, Timestamp_ms: 1744208517450, Avg Intensity: 114.9855, Frame Time: 16.01 ms
Frame: 99, Timestamp_ms: 1744208517472, Avg Intensity: 115.0032, Frame Time: 15.75 ms
Processing complete.
Releasing camera and freeing memory...
Cleanup complete.
```

*(Note: Exact timestamps, average intensity values, and frame times will vary based on lighting conditions, camera, and system load.)*

## Learnings and Observations

*   Integrating OpenCV for camera capture with CUDA for processing is straightforward but requires careful handling of data types and memory transfers.
*   The two-pass reduction strategy works well for summing large arrays (like image pixels) on the GPU.
*   The H->D memory copy per frame is a significant performance limiter in this simple approach.
*   Ensuring the correct camera index and frame format (`CV_8UC3`) is important for reliable operation. Headless debugging can be challenging; clear logging is essential.

## Future Improvements

*   Explore CUDA-OpenCV interoperability (`cv::cuda::GpuMat`) to potentially reduce explicit H->D transfers if processing pipelines become more complex.
*   Use asynchronous memory copies (`cudaMemcpyAsync`) and streams to overlap data transfer and kernel execution.
*   Implement more complex GPU processing beyond average intensity (e.g., feature detection, filtering).
*   Write logs to a file instead of stdout for better data capture.

## Visualization (Optional)

A simple Python script using OpenCV is provided to view the grayscale camera feed directly using CPU processing. This is useful for verifying camera operation and viewing the input visually.

**Prerequisites:**

*   Python 3
*   OpenCV Python bindings (`python3-opencv`): Install using `sudo apt-get install python3-opencv`.
*   A graphical environment accessible on the Jetson Nano (e.g., via VNC, NoMachine, or a connected display).

**Running the Visualization Script:**

1.  Navigate to the day's directory: `cd day030`
2.  Run the script: `python3 view_grayscale.py`
3.  (Optional) Specify a different camera index: `python3 view_grayscale.py 1`

A window will appear showing the live grayscale feed from the camera. Press 'q' in the window to quit.
