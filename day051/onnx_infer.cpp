// day051/onnx_infer.cpp
// Simple TensorRT ONNX Inference Example (Jetson Nano compatible)
// Focus: Demonstrate basic TensorRT workflow for ONNX model inference

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <memory>

// Error checking macro for CUDA
#define CHECK_CUDA(status) \
    do { \
        auto ret = (status); \
        if (ret != 0) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(ret) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
};

std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Failed to open file: " + filename);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) throw std::runtime_error("Failed to read file: " + filename);
    return buffer;
}

int main() {
    Logger logger;

    // 1. Create builder, network, and parser
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (!builder) {
        std::cerr << "Failed to create TensorRT builder." << std::endl;
        return 1;
    }
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));

    // 2. Parse ONNX model
    const std::string onnx_filename = "/home/drboom/cuda-data-sets/mnist.onnx"; // Place your ONNX model here
    std::cout << "Parsing ONNX model: " << onnx_filename << std::endl;
    if (!parser->parseFromFile(onnx_filename.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX model." << std::endl;
        return 1;
    }

    // 3. Build engine
    builder->setMaxBatchSize(1);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1 << 20); // 1 MiB
    std::cout << "Building TensorRT engine..." << std::endl;
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
    if (!engine) {
        std::cerr << "Failed to build engine." << std::endl;
        return 1;
    }
    delete config;

    // 4. Create execution context
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (!context) {
        std::cerr << "Failed to create execution context." << std::endl;
        return 1;
    }

    // 5. Prepare input/output buffers
    int inputIndex = engine->getBindingIndex(engine->getBindingName(0));
    int outputIndex = engine->getBindingIndex(engine->getBindingName(1));
    auto inputDims = engine->getBindingDimensions(inputIndex);
    auto outputDims = engine->getBindingDimensions(outputIndex);

    size_t inputSize = 1;
    for (int i = 0; i < inputDims.nbDims; ++i) inputSize *= inputDims.d[i];
    size_t outputSize = 1;
    for (int i = 0; i < outputDims.nbDims; ++i) outputSize *= outputDims.d[i];

    std::vector<float> inputHost(inputSize, 0.0f); // Dummy input (all zeros)
    std::vector<float> outputHost(outputSize);

    float* inputDevice = nullptr;
    float* outputDevice = nullptr;
    CHECK_CUDA(cudaMalloc(&inputDevice, inputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&outputDevice, outputSize * sizeof(float)));

    // 6. Copy input to device
    CHECK_CUDA(cudaMemcpy(inputDevice, inputHost.data(), inputSize * sizeof(float), cudaMemcpyHostToDevice));

    // 7. Run inference
    void* bindings[2] = {inputDevice, outputDevice};
    std::cout << "Running inference..." << std::endl;
    bool success = context->executeV2(bindings);
    if (!success) {
        std::cerr << "Inference failed." << std::endl;
        return 1;
    }

    // 8. Copy output back to host
    CHECK_CUDA(cudaMemcpy(outputHost.data(), outputDevice, outputSize * sizeof(float), cudaMemcpyDeviceToHost));

    // 9. Print output (e.g., classification scores)
    std::cout << "Output: ";
    for (size_t i = 0; i < outputSize; ++i) {
        std::cout << outputHost[i] << " ";
    }
    std::cout << std::endl;

    // 10. Cleanup
    cudaFree(inputDevice);
    cudaFree(outputDevice);

    std::cout << "Inference complete." << std::endl;
    return 0;
}
