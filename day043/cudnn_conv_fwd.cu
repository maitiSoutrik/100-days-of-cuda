#include <iostream>
#include <vector>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

// Error checking macro for CUDA calls
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Error checking macro for cuDNN calls
#define CHECK_CUDNN_ERROR(call) do { \
    cudnnStatus_t status = call; \
    if (status != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "cuDNN Error at %s:%d - %s\n", __FILE__, __LINE__, cudnnGetErrorString(status)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Helper function to print tensor data (optional, for debugging)
void printTensor(const float* tensor, int n, int c, int h, int w, const char* name) {
    printf("Tensor: %s (Shape: %d, %d, %d, %d)\n", name, n, c, h, w);
    // Print a small subset for brevity
    int count = 0;
    const int max_print = 20;
    for (int i = 0; i < n * c * h * w && count < max_print; ++i) {
        printf("%f ", tensor[i]);
        count++;
    }
    if (count == max_print) printf("...");
    printf("\n\n");
}

int main() {
    // 1. Define Tensor Dimensions (NCHW format)
    int n = 1;      // Batch size
    int c = 3;      // Input channels
    int h = 512;    // Input height (Increased from 224)
    int w = 512;    // Input width (Increased from 224)

    // 2. Define Filter Dimensions (KCRS format)
    int k = 64;     // Output channels (number of filters)
    int r = 3;      // Filter height
    int s = 3;      // Filter width

    // 3. Define Convolution Parameters
    int pad_h = 1;  // Padding height
    int pad_w = 1;  // Padding width
    int stride_h = 1; // Stride height
    int stride_w = 1; // Stride width
    int dilation_h = 1; // Dilation height (not used in basic conv)
    int dilation_w = 1; // Dilation width (not used in basic conv)

    // 4. Calculate Output Dimensions
    int out_n, out_c, out_h, out_w;
    // Formula: O = (I - F + 2P) / S + 1
    out_n = n;
    out_c = k;
    out_h = (h - r + 2 * pad_h) / stride_h + 1;
    out_w = (w - s + 2 * pad_w) / stride_w + 1;

    std::cout << "Input Tensor:  (" << n << ", " << c << ", " << h << ", " << w << ")" << std::endl;
    std::cout << "Filter Tensor: (" << k << ", " << c << ", " << r << ", " << s << ")" << std::endl; // KCRS filter format for cuDNN
    std::cout << "Output Tensor: (" << out_n << ", " << out_c << ", " << out_h << ", " << out_w << ")" << std::endl;

    // 5. Initialize Host Data
    size_t input_size = n * c * h * w;
    size_t filter_size = k * c * r * s; // KCRS
    size_t output_size = out_n * out_c * out_h * out_w;

    std::vector<float> h_input(input_size);
    std::vector<float> h_filter(filter_size);
    std::vector<float> h_output_cudnn(output_size);

    // Fill with some data (e.g., sequential or random)
    for (size_t i = 0; i < input_size; ++i) h_input[i] = static_cast<float>(rand()) / RAND_MAX * 0.1f; // Small random values
    for (size_t i = 0; i < filter_size; ++i) h_filter[i] = static_cast<float>(rand()) / RAND_MAX * 0.01f; // Smaller random values


    // 6. Allocate Device Memory
    float *d_input, *d_filter, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, input_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_filter, filter_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, output_size * sizeof(float)));

    // 7. Copy Data to Device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_filter, h_filter.data(), filter_size * sizeof(float), cudaMemcpyHostToDevice));

    // --- cuDNN Setup ---

    // 8. Create cuDNN Handle
    cudnnHandle_t cudnnHandle;
    CHECK_CUDNN_ERROR(cudnnCreate(&cudnnHandle));

    // 9. Create Tensor Descriptors (Input and Output)
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&inputDesc));
    CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&outputDesc));
    CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(inputDesc,
                                                CUDNN_TENSOR_NCHW, // Format
                                                CUDNN_DATA_FLOAT,  // Data type
                                                n, c, h, w));     // Dimensions
    CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(outputDesc,
                                                CUDNN_TENSOR_NCHW, // Format
                                                CUDNN_DATA_FLOAT,  // Data type
                                                out_n, out_c, out_h, out_w)); // Dimensions

    // 10. Create Filter Descriptor
    cudnnFilterDescriptor_t filterDesc;
    CHECK_CUDNN_ERROR(cudnnCreateFilterDescriptor(&filterDesc));
    CHECK_CUDNN_ERROR(cudnnSetFilter4dDescriptor(filterDesc,
                                                CUDNN_DATA_FLOAT,  // Data type
                                                CUDNN_TENSOR_NCHW, // Format (cuDNN uses NCHW interpretation internally for filters too)
                                                k, c, r, s));     // Dimensions (KCRS order)

    // 11. Create Convolution Descriptor
    cudnnConvolutionDescriptor_t convDesc;
    CHECK_CUDNN_ERROR(cudnnCreateConvolutionDescriptor(&convDesc));
    CHECK_CUDNN_ERROR(cudnnSetConvolution2dDescriptor(convDesc,
                                                      pad_h, pad_w,       // Padding
                                                      stride_h, stride_w, // Stride
                                                      dilation_h, dilation_w, // Dilation
                                                      CUDNN_CONVOLUTION, // Mode (vs CUDNN_CROSS_CORRELATION)
                                                      CUDNN_DATA_FLOAT)); // Compute type

    // --- Convolution Forward Pass ---

    // 12. Choose Convolution Algorithm
    cudnnConvolutionFwdAlgo_t algo;
    // Let cuDNN find the fastest algorithm for the given parameters
    int requestedAlgoCount = 1;
    int returnedAlgoCount = 0;
    cudnnConvolutionFwdAlgoPerf_t perfResults;
    CHECK_CUDNN_ERROR(cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle,
                                                          inputDesc,
                                                          filterDesc,
                                                          convDesc,
                                                          outputDesc,
                                                          requestedAlgoCount,
                                                          &returnedAlgoCount,
                                                          &perfResults));
    algo = perfResults.algo;
    std::cout << "cuDNN selected algorithm: " << algo << " (Status: " << perfResults.status << ", Time: " << perfResults.time << " ms, Memory: " << perfResults.memory << " bytes)" << std::endl;

    // 13. Determine Workspace Size
    size_t workspaceSizeBytes = 0;
    CHECK_CUDNN_ERROR(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                              inputDesc,
                                                              filterDesc,
                                                              convDesc,
                                                              outputDesc,
                                                              algo,
                                                              &workspaceSizeBytes));
    std::cout << "Workspace size: " << workspaceSizeBytes << " bytes" << std::endl;

    // 14. Allocate Workspace Memory
    void* d_workspace = nullptr;
    if (workspaceSizeBytes > 0) {
        CHECK_CUDA_ERROR(cudaMalloc(&d_workspace, workspaceSizeBytes));
    }

    // 15. Execute Convolution Forward Pass
    float alpha = 1.0f; // Scaling factor for input*filter
    float beta = 0.0f;  // Scaling factor for existing output (0 means overwrite)

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaEventRecord(start));

    CHECK_CUDNN_ERROR(cudnnConvolutionForward(cudnnHandle,
                                              &alpha,         // Scaling factor for result
                                              inputDesc,      // Input tensor descriptor
                                              d_input,        // Input data
                                              filterDesc,     // Filter descriptor
                                              d_filter,       // Filter data
                                              convDesc,       // Convolution descriptor
                                              algo,           // Algorithm choice
                                              d_workspace,    // Workspace buffer
                                              workspaceSizeBytes, // Workspace size
                                              &beta,          // Scaling factor for destination tensor
                                              outputDesc,     // Output tensor descriptor
                                              d_output));     // Output data

    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "cuDNN Convolution Forward Time: " << milliseconds << " ms" << std::endl;

    // 16. Copy Result Back to Host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_cudnn.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));

    // 17. (Optional) Verification - Compare with a known result or CPU implementation if available
    // For this example, we'll just print a small part of the output
    printTensor(h_output_cudnn.data(), out_n, out_c, out_h, out_w, "Output (cuDNN)");

    // --- Cleanup ---
    if (d_workspace) CHECK_CUDA_ERROR(cudaFree(d_workspace));
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_filter));
    CHECK_CUDA_ERROR(cudaFree(d_output));

    CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(inputDesc));
    CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(outputDesc));
    CHECK_CUDNN_ERROR(cudnnDestroyFilterDescriptor(filterDesc));
    CHECK_CUDNN_ERROR(cudnnDestroyConvolutionDescriptor(convDesc));
    CHECK_CUDNN_ERROR(cudnnDestroy(cudnnHandle));

    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    std::cout << "cuDNN convolution forward pass completed successfully." << std::endl;

    return 0;
}
