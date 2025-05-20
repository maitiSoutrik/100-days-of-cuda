# Day 71: Group Normalization Forward Pass

## Overview

This project implements the forward pass of Group Normalization in CUDA. Group Normalization (GN) is a normalization technique that divides channels into groups and computes the mean and variance for normalization within each group. It is independent of batch size, making it a good alternative to Batch Normalization for tasks with small or variable batch sizes.

## Implementation Details

The CUDA kernel implements the Group Normalization forward pass.
The `group_norm_main.cu` executable now uses a USB camera as input via OpenCV.
A single frame is captured, resized, and optionally converted to grayscale. This frame (N=1) is then processed by the Group Normalization CUDA kernel.

Input tensor dimensions (N, C, H, W):
- N: Batch size (1 for camera input)
- C: Number of channels (1 for grayscale, 3 for color from camera)
- H: Height of the (resized) camera frame
- W: Width of the (resized) camera frame

The channels are divided into G groups. Normalization is performed independently for each group.

The steps for each group in the sample are:
1. Calculate the mean of the elements in the group.
2. Calculate the variance of the elements in the group.
3. Normalize the elements using the calculated mean and variance.
4. Apply learnable scale (gamma) and shift (beta) parameters.

The program saves the processed input frame as `processed_input.png` and the group-normalized output (first channel) as `group_norm_output.png` in the execution directory. It no longer displays images using `cv::imshow` to ensure compatibility with headless environments.

## Key CUDA Features Used

- CUDA Kernels for parallel computation of mean, variance, and normalization.
- Shared memory for efficient per-group reductions (sum and sum of squares).
- Atomic operations for accumulation in shared memory.

## Dependencies
- CUDA Toolkit
- CMake (>=3.18)
- Google Test (for `day071_group_norm_test`)
- **OpenCV** (for camera input in `day071_group_norm_main`)

## Performance Considerations

- Memory access patterns (coalescing) in the kernel.
- Efficiency of parallel reduction for mean and variance.
- Overhead of OpenCV camera capture and preprocessing.
- **CPU vs. GPU Performance:** For small input sizes (like a single 240x320 grayscale image), the CPU implementation might outperform the GPU version due to CUDA kernel launch overheads and data transfer times. GPUs typically show significant speedups for much larger batch sizes or image dimensions where their massive parallelism can be fully exploited. The Jetson Nano, being an embedded GPU, also has performance characteristics different from discrete datacenter GPUs.

## Building and Running

Ensure OpenCV is installed and discoverable by CMake. The program is designed for headless execution.
To build the project, navigate to the `build` directory and run CMake and Make:

```bash
mkdir -p build
cd build
cmake ..
make group_norm_main
./day071/group_norm_main
```

To run tests:
```bash
# Ensure tests are built
make group_norm_test
ctest --output-on-failure -R Day71_GroupNormTest
```

## Execution Results

Below are sample execution logs from running on a Jetson Nano.

**Main Program Output (`./day071/group_norm_main`):**
```
[ WARN:0@0.796] global cap_gstreamer.cpp:1728 open OpenCV | GStreamer warning: Cannot query video position: status=0, value=-1, duration=-1
Camera opened successfully.
Frame captured: 1280x720 Channels: 3
Converted to grayscale.
Input tensor prepared: N=1 C=1 H=240 W=320
Running GPU Group Normalization...
GPU Execution Time: 76.3077 ms

Running CPU Group Normalization for verification...
CPU Execution Time: 0.432 ms

Verification Successful: GPU and CPU results match.
Processed input frame saved to processed_input.png
Group normalized output saved to group_norm_output.png
```
*(Note: Actual GPU/CPU times may vary. The GStreamer warning is common and usually doesn't affect functionality if the camera works.)*

**Test Program Output (`./day071/group_norm_test`):**
```
[==========] Running 7 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 7 tests from GroupNormTest
[ RUN      ] GroupNormTest.BasicTest
[       OK ] GroupNormTest.BasicTest (68 ms)
[ RUN      ] GroupNormTest.SingleGroup
[       OK ] GroupNormTest.SingleGroup (1 ms)
[ RUN      ] GroupNormTest.GroupsEqualToChannels
[       OK ] GroupNormTest.GroupsEqualToChannels (1 ms)
[ RUN      ] GroupNormTest.LargerDimensions
[       OK ] GroupNormTest.LargerDimensions (2 ms)
[ RUN      ] GroupNormTest.NonUnitGammaBeta
[       OK ] GroupNormTest.NonUnitGammaBeta (1 ms)
[ RUN      ] GroupNormTest.MinimalDimensions
[       OK ] GroupNormTest.MinimalDimensions (1 ms)
[ RUN      ] GroupNormTest.MinimalDimensionsInstanceNormLike
[       OK ] GroupNormTest.MinimalDimensionsInstanceNormLike (1 ms)
[----------] 7 tests from GroupNormTest (76 ms total)

[----------] Global test environment tear-down
[==========] 7 tests from 1 test suite ran. (76 ms total)
[  PASSED  ] 7 tests.
```
The saved images `processed_input.png` and `group_norm_output.png` would also be generated in the execution directory of `group_norm_main`.

## Learnings and Observations

- **Floating-Point Precision:** Minor differences between CPU and GPU floating-point calculations can occur due to different operation ordering or hardware specifics. This might necessitate adjusting tolerances in tests. The `GroupNormTest.LargerDimensions` test case, in particular, required a slightly higher specific tolerance (`1e-3f`) due to these accumulated differences with larger tensor sizes, while other tests pass with a tighter tolerance (`5e-4f`).
- **GPU Performance on Small Data:** For the relatively small input size of a single camera frame processed by `group_norm_main`, the CPU can outperform the GPU. This is expected as CUDA kernel launch overheads and the limited parallelism exploited for small data can negate GPU benefits. The GPU version is expected to scale better with larger N (batch size) or larger C, H, W dimensions.
- **Headless Operation:** Ensuring compatibility with headless environments like the Jetson Nano (when not connected to a display) requires removing direct GUI calls (e.g., `cv::imshow`) and opting for saving outputs to files or other non-GUI feedback.

(Further learnings to be filled in after more extensive testing)

## (Optional) Future Improvements

- Implement backward pass for Group Normalization.
- Optimize kernel performance further.

## (Optional) References

- Group Normalization Paper: [https://arxiv.org/abs/1803.08494](https://arxiv.org/abs/1803.08494)
