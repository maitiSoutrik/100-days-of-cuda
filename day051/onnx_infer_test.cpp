// day051/onnx_infer_test.cpp
// Google Test for TensorRT ONNX Inference

#include <gtest/gtest.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <iostream>
#include <cassert>

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
};

#define CHECK_CUDA(status) \
    do { \
        auto ret = (status); \
        ASSERT_EQ(ret, 0) << "CUDA Error: " << cudaGetErrorString(ret); \
    } while (0)

TEST(OnnxInfer, OutputShapeAndSuccess) {
    Logger logger;
    const std::string onnx_filename = "/home/drboom/cuda-data-sets/mnist.onnx";
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    ASSERT_TRUE(builder != nullptr);

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);

    ASSERT_TRUE(parser->parseFromFile(onnx_filename.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)));

    builder->setMaxBatchSize(1);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1 << 20);
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    ASSERT_TRUE(engine != nullptr);
    config->destroy();

    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    ASSERT_TRUE(context != nullptr);

    int inputIndex = engine->getBindingIndex(engine->getBindingName(0));
    int outputIndex = engine->getBindingIndex(engine->getBindingName(1));
    auto inputDims = engine->getBindingDimensions(inputIndex);
    auto outputDims = engine->getBindingDimensions(outputIndex);

    size_t inputSize = 1;
    for (int i = 0; i < inputDims.nbDims; ++i) inputSize *= inputDims.d[i];
    size_t outputSize = 1;
    for (int i = 0; i < outputDims.nbDims; ++i) outputSize *= outputDims.d[i];

    std::vector<float> inputHost(inputSize, 0.0f);
    std::vector<float> outputHost(outputSize);

    float* inputDevice = nullptr;
    float* outputDevice = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&inputDevice, inputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&outputDevice, outputSize * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(inputDevice, inputHost.data(), inputSize * sizeof(float), cudaMemcpyHostToDevice));

    void* bindings[2] = {inputDevice, outputDevice};
    bool success = context->executeV2(bindings);
    ASSERT_TRUE(success);

    CHECK_CUDA(cudaMemcpy(outputHost.data(), outputDevice, outputSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Test: Output size should be 10 for MNIST
    ASSERT_EQ(outputHost.size(), 10);

    cudaFree(inputDevice);
    cudaFree(outputDevice);
    context->destroy();
    engine->destroy();
    parser->destroy();
    network->destroy();
    builder->destroy();
}
