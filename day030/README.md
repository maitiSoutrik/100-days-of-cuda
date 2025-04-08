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

The program will output lines similar to the following (actual output from Jetson Nano):

```
[ WARN:0@0.862] global cap_gstreamer.cpp:1728 open OpenCV | GStreamer warning: Cannot query video position: status=0, value=-1, duration=-1
Camera opened successfully. Frame dimensions: 1280 x 720 (921600 pixels)
Starting frame capture and processing for 100 frames...
Frame: 0, Timestamp_ms: 1744151153810, Avg Intensity: 133.5533, Frame Time: 21.71 ms
Frame: 1, Timestamp_ms: 1744151153884, Avg Intensity: 130.6042, Frame Time: 73.68 ms
Frame: 2, Timestamp_ms: 1744151153984, Avg Intensity: 130.6792, Frame Time: 100.43 ms
Frame: 3, Timestamp_ms: 1744151154082, Avg Intensity: 130.7072, Frame Time: 98.12 ms
Frame: 4, Timestamp_ms: 1744151154183, Avg Intensity: 127.5448, Frame Time: 100.51 ms
Frame: 5, Timestamp_ms: 1744151154288, Avg Intensity: 127.5394, Frame Time: 105.16 ms
Frame: 6, Timestamp_ms: 1744151154385, Avg Intensity: 127.5250, Frame Time: 96.51 ms
Frame: 7, Timestamp_ms: 1744151154489, Avg Intensity: 127.5336, Frame Time: 104.23 ms
Frame: 8, Timestamp_ms: 1744151154583, Avg Intensity: 127.5530, Frame Time: 93.95 ms
Frame: 9, Timestamp_ms: 1744151154679, Avg Intensity: 127.5513, Frame Time: 95.79 ms
Frame: 10, Timestamp_ms: 1744151154784, Avg Intensity: 127.5546, Frame Time: 105.05 ms
Frame: 11, Timestamp_ms: 1744151154880, Avg Intensity: 127.5777, Frame Time: 95.68 ms
Frame: 12, Timestamp_ms: 1744151154984, Avg Intensity: 127.5855, Frame Time: 104.60 ms
Frame: 13, Timestamp_ms: 1744151155079, Avg Intensity: 127.6000, Frame Time: 94.88 ms
Frame: 14, Timestamp_ms: 1744151155181, Avg Intensity: 127.6019, Frame Time: 101.35 ms
Frame: 15, Timestamp_ms: 1744151155283, Avg Intensity: 127.6048, Frame Time: 102.15 ms
Frame: 16, Timestamp_ms: 1744151155382, Avg Intensity: 127.5995, Frame Time: 98.56 ms
Frame: 17, Timestamp_ms: 1744151155486, Avg Intensity: 127.6131, Frame Time: 104.28 ms
Frame: 18, Timestamp_ms: 1744151155582, Avg Intensity: 127.6262, Frame Time: 96.16 ms
Frame: 19, Timestamp_ms: 1744151155688, Avg Intensity: 127.6291, Frame Time: 105.75 ms
Frame: 20, Timestamp_ms: 1744151155787, Avg Intensity: 127.6460, Frame Time: 99.02 ms
Frame: 21, Timestamp_ms: 1744151155884, Avg Intensity: 127.6516, Frame Time: 96.78 ms
Frame: 22, Timestamp_ms: 1744151155986, Avg Intensity: 127.6542, Frame Time: 101.88 ms
Frame: 23, Timestamp_ms: 1744151156087, Avg Intensity: 127.6568, Frame Time: 101.65 ms
Frame: 24, Timestamp_ms: 1744151156187, Avg Intensity: 127.6726, Frame Time: 99.57 ms
Frame: 25, Timestamp_ms: 1744151156285, Avg Intensity: 127.6540, Frame Time: 98.25 ms
Frame: 26, Timestamp_ms: 1744151156384, Avg Intensity: 127.6548, Frame Time: 99.04 ms
Frame: 27, Timestamp_ms: 1744151156484, Avg Intensity: 127.6675, Frame Time: 99.35 ms
Frame: 28, Timestamp_ms: 1744151156585, Avg Intensity: 127.6866, Frame Time: 101.09 ms
Frame: 29, Timestamp_ms: 1744151156684, Avg Intensity: 127.6925, Frame Time: 98.82 ms
Frame: 30, Timestamp_ms: 1744151156784, Avg Intensity: 127.6973, Frame Time: 99.65 ms
Frame: 31, Timestamp_ms: 1744151156884, Avg Intensity: 127.7044, Frame Time: 100.15 ms
Frame: 32, Timestamp_ms: 1744151156980, Avg Intensity: 127.7237, Frame Time: 96.41 ms
Frame: 33, Timestamp_ms: 1744151157084, Avg Intensity: 127.7332, Frame Time: 103.69 ms
Frame: 34, Timestamp_ms: 1744151157181, Avg Intensity: 127.7304, Frame Time: 96.85 ms
Frame: 35, Timestamp_ms: 1744151157285, Avg Intensity: 127.7373, Frame Time: 103.84 ms
Frame: 36, Timestamp_ms: 1744151157383, Avg Intensity: 127.7487, Frame Time: 98.17 ms
Frame: 37, Timestamp_ms: 1744151157487, Avg Intensity: 127.7693, Frame Time: 103.84 ms
Frame: 38, Timestamp_ms: 1744151157583, Avg Intensity: 127.7728, Frame Time: 96.20 ms
Frame: 39, Timestamp_ms: 1744151157684, Avg Intensity: 127.7943, Frame Time: 101.16 ms
Frame: 40, Timestamp_ms: 1744151157784, Avg Intensity: 127.8206, Frame Time: 99.63 ms
Frame: 41, Timestamp_ms: 1744151157884, Avg Intensity: 127.8262, Frame Time: 99.85 ms
Frame: 42, Timestamp_ms: 1744151157982, Avg Intensity: 127.8493, Frame Time: 97.75 ms
Frame: 43, Timestamp_ms: 1744151158089, Avg Intensity: 127.8519, Frame Time: 106.98 ms
Frame: 44, Timestamp_ms: 1744151158189, Avg Intensity: 127.8548, Frame Time: 100.46 ms
Frame: 45, Timestamp_ms: 1744151158284, Avg Intensity: 127.8716, Frame Time: 95.10 ms
Frame: 46, Timestamp_ms: 1744151158396, Avg Intensity: 127.8868, Frame Time: 111.64 ms
Frame: 47, Timestamp_ms: 1744151158485, Avg Intensity: 127.8990, Frame Time: 88.66 ms
Frame: 48, Timestamp_ms: 1744151158584, Avg Intensity: 127.9302, Frame Time: 99.13 ms
Frame: 49, Timestamp_ms: 1744151158684, Avg Intensity: 127.9442, Frame Time: 99.51 ms
Frame: 50, Timestamp_ms: 1744151158782, Avg Intensity: 127.9378, Frame Time: 97.91 ms
Frame: 51, Timestamp_ms: 1744151158883, Avg Intensity: 127.9671, Frame Time: 100.79 ms
Frame: 52, Timestamp_ms: 1744151158980, Avg Intensity: 127.9718, Frame Time: 97.88 ms
Frame: 53, Timestamp_ms: 1744151159084, Avg Intensity: 127.9897, Frame Time: 103.04 ms
Frame: 54, Timestamp_ms: 1744151159185, Avg Intensity: 128.0068, Frame Time: 101.11 ms
Frame: 55, Timestamp_ms: 1744151159284, Avg Intensity: 128.0310, Frame Time: 98.79 ms
Frame: 56, Timestamp_ms: 1744151159385, Avg Intensity: 128.0496, Frame Time: 100.93 ms
Frame: 57, Timestamp_ms: 1744151159489, Avg Intensity: 128.0725, Frame Time: 104.08 ms
Frame: 58, Timestamp_ms: 1744151159587, Avg Intensity: 128.0851, Frame Time: 98.22 ms
Frame: 59, Timestamp_ms: 1744151159684, Avg Intensity: 128.1121, Frame Time: 96.66 ms
Frame: 60, Timestamp_ms: 1744151159786, Avg Intensity: 128.1357, Frame Time: 101.92 ms
Frame: 61, Timestamp_ms: 1744151159889, Avg Intensity: 128.1308, Frame Time: 103.55 ms
Frame: 62, Timestamp_ms: 1744151159979, Avg Intensity: 128.1515, Frame Time: 90.07 ms
Frame: 63, Timestamp_ms: 1744151160085, Avg Intensity: 128.1821, Frame Time: 105.61 ms
Frame: 64, Timestamp_ms: 1744151160185, Avg Intensity: 128.1965, Frame Time: 99.73 ms
Frame: 65, Timestamp_ms: 1744151160286, Avg Intensity: 128.2035, Frame Time: 100.86 ms
Frame: 66, Timestamp_ms: 1744151160384, Avg Intensity: 128.2242, Frame Time: 97.85 ms
Frame: 67, Timestamp_ms: 1744151160486, Avg Intensity: 128.2471, Frame Time: 102.29 ms
Frame: 68, Timestamp_ms: 1744151160586, Avg Intensity: 128.2398, Frame Time: 100.07 ms
Frame: 69, Timestamp_ms: 1744151160684, Avg Intensity: 128.2611, Frame Time: 98.35 ms
Frame: 70, Timestamp_ms: 1744151160786, Avg Intensity: 128.2634, Frame Time: 101.36 ms
Frame: 71, Timestamp_ms: 1744151160886, Avg Intensity: 128.2599, Frame Time: 100.35 ms
Frame: 72, Timestamp_ms: 1744151160979, Avg Intensity: 128.2845, Frame Time: 92.99 ms
Frame: 73, Timestamp_ms: 1744151161080, Avg Intensity: 128.2808, Frame Time: 100.37 ms
Frame: 74, Timestamp_ms: 1744151161188, Avg Intensity: 128.2798, Frame Time: 108.24 ms
Frame: 75, Timestamp_ms: 1744151161284, Avg Intensity: 128.2800, Frame Time: 95.77 ms
Frame: 76, Timestamp_ms: 1744151161381, Avg Intensity: 128.2602, Frame Time: 97.42 ms
Frame: 77, Timestamp_ms: 1744151161482, Avg Intensity: 128.2719, Frame Time: 100.45 ms
Frame: 78, Timestamp_ms: 1744151161589, Avg Intensity: 128.2678, Frame Time: 106.88 ms
Frame: 79, Timestamp_ms: 1744151161684, Avg Intensity: 128.2690, Frame Time: 95.06 ms
Frame: 80, Timestamp_ms: 1744151161784, Avg Intensity: 128.2690, Frame Time: 99.78 ms
Frame: 81, Timestamp_ms: 1744151161890, Avg Intensity: 128.2617, Frame Time: 105.84 ms
Frame: 82, Timestamp_ms: 1744151161989, Avg Intensity: 128.2587, Frame Time: 99.73 ms
Frame: 83, Timestamp_ms: 1744151162090, Avg Intensity: 128.2712, Frame Time: 100.92 ms
Frame: 84, Timestamp_ms: 1744151162187, Avg Intensity: 128.2791, Frame Time: 96.62 ms
Frame: 85, Timestamp_ms: 1744151162289, Avg Intensity: 128.2802, Frame Time: 101.45 ms
Frame: 86, Timestamp_ms: 1744151162388, Avg Intensity: 128.2834, Frame Time: 99.21 ms
Frame: 87, Timestamp_ms: 1744151162487, Avg Intensity: 128.2915, Frame Time: 99.52 ms
Frame: 88, Timestamp_ms: 1744151162584, Avg Intensity: 128.2799, Frame Time: 96.82 ms
Frame: 89, Timestamp_ms: 1744151162684, Avg Intensity: 128.2754, Frame Time: 100.06 ms
Frame: 90, Timestamp_ms: 1744151162785, Avg Intensity: 128.2706, Frame Time: 100.25 ms
Frame: 91, Timestamp_ms: 1744151162886, Avg Intensity: 128.2751, Frame Time: 101.81 ms
Frame: 92, Timestamp_ms: 1744151162980, Avg Intensity: 128.2827, Frame Time: 93.55 ms
Frame: 93, Timestamp_ms: 1744151163083, Avg Intensity: 128.2795, Frame Time: 103.22 ms
Frame: 94, Timestamp_ms: 1744151163186, Avg Intensity: 128.2670, Frame Time: 102.19 ms
Frame: 95, Timestamp_ms: 1744151163287, Avg Intensity: 128.2693, Frame Time: 101.38 ms
Frame: 96, Timestamp_ms: 1744151163388, Avg Intensity: 128.2723, Frame Time: 100.93 ms
Frame: 97, Timestamp_ms: 1744151163484, Avg Intensity: 128.2624, Frame Time: 96.42 ms
Frame: 98, Timestamp_ms: 1744151163584, Avg Intensity: 128.2670, Frame Time: 99.33 ms
Frame: 99, Timestamp_ms: 1744151163682, Avg Intensity: 128.2667, Frame Time: 98.39 ms
Processing complete.
Releasing camera and freeing memory...
Cleanup complete.
```

*(Note: The GStreamer warning is common and often doesn't affect functionality. Frame times will vary based on system load and specific camera/Jetson performance.)*

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
