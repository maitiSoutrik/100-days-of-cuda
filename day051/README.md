# Day 51: Exploring TensorRT (Simple ONNX Inference)

## Overview

This day demonstrates how to use NVIDIA TensorRT to optimize and run deep learning inference on NVIDIA GPUs, focusing on edge devices like the Jetson Nano. The goal is to take a simple pre-trained ONNX model (e.g., an MNIST classifier) and use the TensorRT C++ API to:
1. Build an optimized engine from the ONNX file.
2. Create an execution context.
3. Allocate input/output buffers on the GPU.
4. Run inference on sample input data.
5. Retrieve and interpret the output.

## Implementation Details

- The code (`onnx_infer.cpp`) uses the TensorRT C++ API and the ONNX parser to load an ONNX model, build an inference engine, and execute inference.
- The workflow:
    - Loads the ONNX model (default: `mnist.onnx`).
    - Builds a TensorRT engine with explicit batch.
    - Allocates device memory for input and output.
    - Runs inference on dummy input (all zeros for demonstration).
    - Retrieves and prints the output scores.
- Error checking is included for CUDA and TensorRT API calls.
- The code is designed for Jetson Nano (Compute Capability 5.3) and requires TensorRT and CUDA to be installed.

**Note:** You must provide a compatible ONNX model (e.g., download a small MNIST classifier from the [ONNX Model Zoo](https://github.com/onnx/models)) and place it as `mnist.onnx` in this directory.

## Key CUDA Concepts

- TensorRT workflow: builder, network, parser, engine, execution context.
- ONNX model parsing and optimization.
- GPU memory management for inference.
- CUDA error checking.

## Performance Considerations

- TensorRT provides significant speedup for deep learning inference on NVIDIA hardware by optimizing the computation graph and leveraging GPU acceleration.
- The example uses a small workspace and batch size for demonstration; for real applications, tune these parameters for best performance.
- For benchmarking, compare inference time with and without TensorRT, and with CPU-based inference if possible.

## Building and Running

**Requirements (on Jetson Nano or compatible environment):**
- CUDA Toolkit
- TensorRT SDK (with C++ API and ONNX parser)
- CMake >= 3.10
- A compatible ONNX model (e.g., `mnist.onnx`)

**Steps:**
```bash
# From the project root
cd day051
# Place your ONNX model as mnist.onnx in this directory
mkdir build && cd build
cmake ..
make
./onnx_infer
```

## Execution Results

### Inference Binary Output

```
Parsing ONNX model: /home/drboom/cuda-data-sets/mnist.onnx
----------------------------------------------------------------
Input filename:   /home/drboom/cuda-data-sets/mnist.onnx
ONNX IR version:  0.0.3
Opset version:    8
Producer name:    CNTK
Producer version: 2.5.1
Domain:           ai.cntk
Model version:    1
Doc string:
----------------------------------------------------------------
[TensorRT] onnx2trt_utils.cpp:220: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
Building TensorRT engine...
Running inference 100 times for benchmarking...
Average inference time over 100 runs: 0.493875 ms
Output: -0.044856 0.00779166 0.0681008 0.0299937 -0.12641 0.140219 -0.0552849 -0.0493838 0.0843221 -0.0545404
Inference complete
```

*Interpret the output as class scores or probabilities. For MNIST, the highest value indicates the predicted digit.*

### Google Test Output

```
Running main() from /home/drboom/git_repos/100-days-of-cuda/build/_deps/googletest-src/googletest/src/gtest_main.cc
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from OnnxInfer
[ RUN      ] OnnxInfer.OutputShapeAndSuccess
----------------------------------------------------------------
Input filename:   /home/drboom/cuda-data-sets/mnist.onnx
ONNX IR version:  0.0.3
Opset version:    8
Producer name:    CNTK
Producer version: 2.5.1
Domain:           ai.cntk
Model version:    1
Doc string:
----------------------------------------------------------------
[TensorRT] onnx2trt_utils.cpp:220: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
[       OK ] OnnxInfer.OutputShapeAndSuccess (5274 ms)
[----------] 1 test from OnnxInfer (5274 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (5274 ms total)
[  PASSED  ] 1 test.
```

## Testing

Google Test is integrated for automated validation in CI and local builds. The test:
- Loads the ONNX model and runs inference.
- Asserts that inference completes successfully.
- Checks that the output vector has the correct size (10 for MNIST).

**To run the test manually:**
```bash
cd build
./day051/onnx_infer_test
```
Or run all tests with:
```bash
ctest
```

## Learnings and Observations

- TensorRT's workflow is modular: parsing, building, and executing are distinct steps.
- ONNX format enables interoperability between frameworks and deployment tools.
- Proper error checking is crucial for debugging TensorRT and CUDA code.
- Jetson Nano can efficiently run optimized deep learning models with low latency.
- Automated testing with Google Test ensures inference code is robust and CI-friendly.

## Future Improvements

- Use real input data (e.g., actual MNIST images) for inference.
- Benchmark inference time and compare with CPU or non-TensorRT GPU inference.
- Explore INT8/FP16 precision for further optimization.
- Batch inference and stream processing.

## References

- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- [ONNX Model Zoo](https://github.com/onnx/models)
- [TensorRT C++ Samples](https://github.com/NVIDIA/TensorRT/tree/main/samples)
