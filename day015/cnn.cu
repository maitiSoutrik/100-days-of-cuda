#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA_ERROR(call) \
{ \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at line %d\n", cudaGetErrorString(error), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// Dimensions for our example
#define INPUT_HEIGHT 28
#define INPUT_WIDTH 28
#define INPUT_CHANNELS 1
#define FILTER_SIZE 3
#define NUM_FILTERS 16
#define STRIDE 1
#define PADDING 1
#define POOL_SIZE 2
#define POOL_STRIDE 2

// Calculate output dimensions after convolution
#define OUTPUT_HEIGHT ((INPUT_HEIGHT + 2 * PADDING - FILTER_SIZE) / STRIDE + 1)
#define OUTPUT_WIDTH ((INPUT_WIDTH + 2 * PADDING - FILTER_SIZE) / STRIDE + 1)

// Calculate output dimensions after pooling
#define POOL_OUTPUT_HEIGHT (OUTPUT_HEIGHT / POOL_STRIDE)
#define POOL_OUTPUT_WIDTH (OUTPUT_WIDTH / POOL_STRIDE)

// Utility function to initialize data
void initializeData(float *data, int size, float min, float max) {
    for (int i = 0; i < size; i++) {
        data[i] = min + (max - min) * (float)rand() / RAND_MAX;
    }
}

// Utility function to check results
void checkResults(float *host, float *device, int size, const char *message) {
    float *result = (float *)malloc(size * sizeof(float));
    CHECK_CUDA_ERROR(cudaMemcpy(result, device, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    float maxError = 0.0f;
    for (int i = 0; i < size; i++) {
        maxError = fmax(maxError, fabs(result[i] - host[i]));
    }
    
    printf("%s: Max Error = %f\n", message, maxError);
    free(result);
}

// ReLU activation function kernel
__global__ void reluActivationKernel(float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// ReLU derivative kernel for backpropagation
__global__ void reluDerivativeKernel(float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (input[idx] > 0.0f) ? 1.0f : 0.0f;
    }
}

// Im2Col kernel for unrolling input data for convolution
__global__ void im2colKernel(
    float *input, float *output,
    int inputHeight, int inputWidth, int channels,
    int kernelHeight, int kernelWidth,
    int padHeight, int padWidth,
    int strideHeight, int strideWidth,
    int outputHeight, int outputWidth
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < outputHeight * outputWidth * kernelHeight * kernelWidth * channels) {
        // Calculate indices
        int colIdx = idx;
        int channelCol = colIdx % channels;
        colIdx /= channels;
        int kernelWidthCol = colIdx % kernelWidth;
        colIdx /= kernelWidth;
        int kernelHeightCol = colIdx % kernelHeight;
        colIdx /= kernelHeight;
        int widthCol = colIdx % outputWidth;
        colIdx /= outputWidth;
        int heightCol = colIdx;
        
        // Calculate input indices with padding
        int heightOffset = heightCol * strideHeight - padHeight + kernelHeightCol;
        int widthOffset = widthCol * strideWidth - padWidth + kernelWidthCol;
        
        // Set output value
        if (heightOffset >= 0 && heightOffset < inputHeight && 
            widthOffset >= 0 && widthOffset < inputWidth) {
            output[idx] = input[(channelCol * inputHeight + heightOffset) * inputWidth + widthOffset];
        } else {
            output[idx] = 0.0f;
        }
    }
}

// Matrix multiplication kernel for convolution
__global__ void matrixMultiplyKernel(
    float *A, float *B, float *C,
    int numARows, int numAColumns,
    int numBRows, int numBColumns,
    int numCRows, int numCColumns
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < numCRows && col < numCColumns) {
        float sum = 0.0f;
        for (int k = 0; k < numAColumns; k++) {
            sum += A[row * numAColumns + k] * B[k * numBColumns + col];
        }
        C[row * numCColumns + col] = sum;
    }
}

// Max pooling kernel
__global__ void maxPoolingKernel(
    float *input, float *output, int *indices,
    int inputHeight, int inputWidth, int channels,
    int poolHeight, int poolWidth,
    int strideHeight, int strideWidth,
    int outputHeight, int outputWidth
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < outputHeight * outputWidth * channels) {
        // Calculate indices
        int outputIdx = idx;
        int channelIdx = outputIdx % channels;
        outputIdx /= channels;
        int outputWidthIdx = outputIdx % outputWidth;
        outputIdx /= outputWidth;
        int outputHeightIdx = outputIdx;
        
        // Calculate input region
        int inputHeightStart = outputHeightIdx * strideHeight;
        int inputWidthStart = outputWidthIdx * strideWidth;
        
        // Find max value in pool window
        float maxVal = -INFINITY;
        int maxIdx = -1;
        
        for (int h = 0; h < poolHeight; h++) {
            for (int w = 0; w < poolWidth; w++) {
                int inputHeightIdx = inputHeightStart + h;
                int inputWidthIdx = inputWidthStart + w;
                
                if (inputHeightIdx < inputHeight && inputWidthIdx < inputWidth) {
                    int inputIdx = (channelIdx * inputHeight + inputHeightIdx) * inputWidth + inputWidthIdx;
                    float val = input[inputIdx];
                    
                    if (val > maxVal) {
                        maxVal = val;
                        maxIdx = inputIdx;
                    }
                }
            }
        }
        
        // Store max value and its index
        output[idx] = maxVal;
        indices[idx] = maxIdx;
    }
}

// Max pooling backpropagation kernel
__global__ void maxPoolingBackpropKernel(
    float *outputGradient, float *inputGradient, int *indices,
    int outputSize, int inputSize
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < outputSize) {
        int inputIdx = indices[idx];
        if (inputIdx >= 0 && inputIdx < inputSize) {
            atomicAdd(&inputGradient[inputIdx], outputGradient[idx]);
        }
    }
}

// Convolution backpropagation kernel for input gradients
__global__ void convolutionBackpropInputKernel(
    float *outputGradient, float *filters, float *inputGradient,
    int outputHeight, int outputWidth, int outputChannels,
    int filterHeight, int filterWidth, int inputChannels,
    int inputHeight, int inputWidth,
    int padHeight, int padWidth,
    int strideHeight, int strideWidth
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < inputHeight * inputWidth * inputChannels) {
        // Calculate indices
        int inputIdx = idx;
        int inputChannelIdx = inputIdx % inputChannels;
        inputIdx /= inputChannels;
        int inputWidthIdx = inputIdx % inputWidth;
        inputIdx /= inputWidth;
        int inputHeightIdx = inputIdx;
        
        float gradient = 0.0f;
        
        // Iterate over all output positions that could have contributed to this input
        for (int outputChannelIdx = 0; outputChannelIdx < outputChannels; outputChannelIdx++) {
            for (int filterHeightIdx = 0; filterHeightIdx < filterHeight; filterHeightIdx++) {
                for (int filterWidthIdx = 0; filterWidthIdx < filterWidth; filterWidthIdx++) {
                    // Calculate corresponding output position
                    int outputHeightIdx = (inputHeightIdx + padHeight - filterHeightIdx) / strideHeight;
                    int outputWidthIdx = (inputWidthIdx + padWidth - filterWidthIdx) / strideWidth;
                    
                    // Check if output position is valid and aligns with stride
                    if (outputHeightIdx >= 0 && outputHeightIdx < outputHeight &&
                        outputWidthIdx >= 0 && outputWidthIdx < outputWidth &&
                        (inputHeightIdx + padHeight - filterHeightIdx) % strideHeight == 0 &&
                        (inputWidthIdx + padWidth - filterWidthIdx) % strideWidth == 0) {
                        
                        // Calculate filter index
                        int filterIdx = ((outputChannelIdx * inputChannels + inputChannelIdx) * filterHeight + filterHeightIdx) * filterWidth + filterWidthIdx;
                        
                        // Calculate output gradient index
                        int outputGradientIdx = (outputChannelIdx * outputHeight + outputHeightIdx) * outputWidth + outputWidthIdx;
                        
                        // Accumulate gradient
                        gradient += outputGradient[outputGradientIdx] * filters[filterIdx];
                    }
                }
            }
        }
        
        // Store input gradient
        inputGradient[idx] = gradient;
    }
}

// Convolution backpropagation kernel for filter gradients
__global__ void multiplyGradientsKernel(float *a, float *b, float *c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] * b[idx];
    }
}

__global__ void convolutionBackpropFilterKernel(
    float *input, float *outputGradient, float *filterGradient,
    int inputHeight, int inputWidth, int inputChannels,
    int filterHeight, int filterWidth,
    int outputHeight, int outputWidth, int outputChannels,
    int padHeight, int padWidth,
    int strideHeight, int strideWidth
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < outputChannels * inputChannels * filterHeight * filterWidth) {
        // Calculate indices
        int filterIdx = idx;
        int filterWidthIdx = filterIdx % filterWidth;
        filterIdx /= filterWidth;
        int filterHeightIdx = filterIdx % filterHeight;
        filterIdx /= filterHeight;
        int inputChannelIdx = filterIdx % inputChannels;
        filterIdx /= inputChannels;
        int outputChannelIdx = filterIdx;
        
        float gradient = 0.0f;
        
        // Iterate over all output positions
        for (int outputHeightIdx = 0; outputHeightIdx < outputHeight; outputHeightIdx++) {
            for (int outputWidthIdx = 0; outputWidthIdx < outputWidth; outputWidthIdx++) {
                // Calculate corresponding input position
                int inputHeightIdx = outputHeightIdx * strideHeight - padHeight + filterHeightIdx;
                int inputWidthIdx = outputWidthIdx * strideWidth - padWidth + filterWidthIdx;
                
                // Check if input position is valid
                if (inputHeightIdx >= 0 && inputHeightIdx < inputHeight &&
                    inputWidthIdx >= 0 && inputWidthIdx < inputWidth) {
                    
                    // Calculate input index
                    int inputIdx = (inputChannelIdx * inputHeight + inputHeightIdx) * inputWidth + inputWidthIdx;
                    
                    // Calculate output gradient index
                    int outputGradientIdx = (outputChannelIdx * outputHeight + outputHeightIdx) * outputWidth + outputWidthIdx;
                    
                    // Accumulate gradient
                    gradient += input[inputIdx] * outputGradient[outputGradientIdx];
                }
            }
        }
        
        // Store filter gradient
        filterGradient[idx] = gradient;
    }
}

// Main function to demonstrate CNN operations
int main() {
    // Seed random number generator
    srand(42);
    
    // Allocate host memory
    float *h_input = (float *)malloc(INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS * sizeof(float));
    float *h_filters = (float *)malloc(NUM_FILTERS * INPUT_CHANNELS * FILTER_SIZE * FILTER_SIZE * sizeof(float));
    float *h_bias = (float *)malloc(NUM_FILTERS * sizeof(float));
    
    // Initialize host data
    initializeData(h_input, INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS, -1.0f, 1.0f);
    initializeData(h_filters, NUM_FILTERS * INPUT_CHANNELS * FILTER_SIZE * FILTER_SIZE, -0.1f, 0.1f);
    initializeData(h_bias, NUM_FILTERS, -0.1f, 0.1f);
    
    // Allocate device memory
    float *d_input, *d_filters, *d_bias;
    float *d_conv_output, *d_relu_output, *d_pool_output;
    int *d_pool_indices;
    
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_input, INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_filters, NUM_FILTERS * INPUT_CHANNELS * FILTER_SIZE * FILTER_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_bias, NUM_FILTERS * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_conv_output, OUTPUT_HEIGHT * OUTPUT_WIDTH * NUM_FILTERS * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_relu_output, OUTPUT_HEIGHT * OUTPUT_WIDTH * NUM_FILTERS * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_pool_output, POOL_OUTPUT_HEIGHT * POOL_OUTPUT_WIDTH * NUM_FILTERS * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_pool_indices, POOL_OUTPUT_HEIGHT * POOL_OUTPUT_WIDTH * NUM_FILTERS * sizeof(int)));
    
    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_filters, h_filters, NUM_FILTERS * INPUT_CHANNELS * FILTER_SIZE * FILTER_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_bias, h_bias, NUM_FILTERS * sizeof(float), cudaMemcpyHostToDevice));
    
    // Allocate memory for im2col
    int im2col_height = FILTER_SIZE * FILTER_SIZE * INPUT_CHANNELS;
    int im2col_width = OUTPUT_HEIGHT * OUTPUT_WIDTH;
    float *d_im2col_data;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_im2col_data, im2col_height * im2col_width * sizeof(float)));
    
    // Define block and grid dimensions
    dim3 im2colBlock(256);
    dim3 im2colGrid((im2col_height * im2col_width + im2colBlock.x - 1) / im2colBlock.x);
    
    dim3 matrixMultBlock(16, 16);
    dim3 matrixMultGrid(
        (im2col_width + matrixMultBlock.x - 1) / matrixMultBlock.x,
        (NUM_FILTERS + matrixMultBlock.y - 1) / matrixMultBlock.y
    );
    
    dim3 reluBlock(256);
    dim3 reluGrid((OUTPUT_HEIGHT * OUTPUT_WIDTH * NUM_FILTERS + reluBlock.x - 1) / reluBlock.x);
    
    dim3 poolBlock(256);
    dim3 poolGrid((POOL_OUTPUT_HEIGHT * POOL_OUTPUT_WIDTH * NUM_FILTERS + poolBlock.x - 1) / poolBlock.x);
    
    // Forward Pass
    printf("Performing forward pass...\n");
    
    // Step 1: Im2Col transformation
    im2colKernel<<<im2colGrid, im2colBlock>>>(
        d_input, d_im2col_data,
        INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS,
        FILTER_SIZE, FILTER_SIZE,
        PADDING, PADDING,
        STRIDE, STRIDE,
        OUTPUT_HEIGHT, OUTPUT_WIDTH
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Step 2: Matrix multiplication (convolution)
    matrixMultiplyKernel<<<matrixMultGrid, matrixMultBlock>>>(
        d_filters, d_im2col_data, d_conv_output,
        NUM_FILTERS, im2col_height,
        im2col_height, im2col_width,
        NUM_FILTERS, im2col_width
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Step 3: ReLU activation
    reluActivationKernel<<<reluGrid, reluBlock>>>(
        d_conv_output, d_relu_output, OUTPUT_HEIGHT * OUTPUT_WIDTH * NUM_FILTERS
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Step 4: Max pooling
    maxPoolingKernel<<<poolGrid, poolBlock>>>(
        d_relu_output, d_pool_output, d_pool_indices,
        OUTPUT_HEIGHT, OUTPUT_WIDTH, NUM_FILTERS,
        POOL_SIZE, POOL_SIZE,
        POOL_STRIDE, POOL_STRIDE,
        POOL_OUTPUT_HEIGHT, POOL_OUTPUT_WIDTH
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Allocate memory for backpropagation
    float *d_pool_gradient, *d_relu_gradient, *d_conv_gradient;
    float *d_filter_gradient, *d_input_gradient;
    
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_pool_gradient, POOL_OUTPUT_HEIGHT * POOL_OUTPUT_WIDTH * NUM_FILTERS * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_relu_gradient, OUTPUT_HEIGHT * OUTPUT_WIDTH * NUM_FILTERS * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_conv_gradient, OUTPUT_HEIGHT * OUTPUT_WIDTH * NUM_FILTERS * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_filter_gradient, NUM_FILTERS * INPUT_CHANNELS * FILTER_SIZE * FILTER_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_input_gradient, INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS * sizeof(float)));
    
    // Initialize gradients for demonstration (normally would come from next layer)
    float *h_pool_gradient = (float *)malloc(POOL_OUTPUT_HEIGHT * POOL_OUTPUT_WIDTH * NUM_FILTERS * sizeof(float));
    initializeData(h_pool_gradient, POOL_OUTPUT_HEIGHT * POOL_OUTPUT_WIDTH * NUM_FILTERS, -0.01f, 0.01f);
    CHECK_CUDA_ERROR(cudaMemcpy(d_pool_gradient, h_pool_gradient, POOL_OUTPUT_HEIGHT * POOL_OUTPUT_WIDTH * NUM_FILTERS * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize other gradients to zero
    CHECK_CUDA_ERROR(cudaMemset(d_relu_gradient, 0, OUTPUT_HEIGHT * OUTPUT_WIDTH * NUM_FILTERS * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_conv_gradient, 0, OUTPUT_HEIGHT * OUTPUT_WIDTH * NUM_FILTERS * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_filter_gradient, 0, NUM_FILTERS * INPUT_CHANNELS * FILTER_SIZE * FILTER_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_input_gradient, 0, INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS * sizeof(float)));
    
    // Backward Pass
    printf("\nPerforming backward pass...\n");
    
    // Step 1: Max pooling backpropagation
    maxPoolingBackpropKernel<<<poolGrid, poolBlock>>>(
        d_pool_gradient, d_relu_gradient, d_pool_indices,
        POOL_OUTPUT_HEIGHT * POOL_OUTPUT_WIDTH * NUM_FILTERS,
        OUTPUT_HEIGHT * OUTPUT_WIDTH * NUM_FILTERS
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Step 2: ReLU backpropagation
    reluDerivativeKernel<<<reluGrid, reluBlock>>>(
        d_relu_output, d_conv_gradient, OUTPUT_HEIGHT * OUTPUT_WIDTH * NUM_FILTERS
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Element-wise multiplication of gradients
    dim3 elementWiseBlock(256);
    dim3 elementWiseGrid((OUTPUT_HEIGHT * OUTPUT_WIDTH * NUM_FILTERS + elementWiseBlock.x - 1) / elementWiseBlock.x);
    
    multiplyGradientsKernel<<<elementWiseGrid, elementWiseBlock>>>(
        d_relu_gradient, d_conv_gradient, d_conv_gradient, OUTPUT_HEIGHT * OUTPUT_WIDTH * NUM_FILTERS
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Step 3: Convolution backpropagation for filters
    dim3 filterGradBlock(256);
    dim3 filterGradGrid((NUM_FILTERS * INPUT_CHANNELS * FILTER_SIZE * FILTER_SIZE + filterGradBlock.x - 1) / filterGradBlock.x);
    
    convolutionBackpropFilterKernel<<<filterGradGrid, filterGradBlock>>>(
        d_input, d_conv_gradient, d_filter_gradient,
        INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS,
        FILTER_SIZE, FILTER_SIZE,
        OUTPUT_HEIGHT, OUTPUT_WIDTH, NUM_FILTERS,
        PADDING, PADDING,
        STRIDE, STRIDE
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Step 4: Convolution backpropagation for input
    dim3 inputGradBlock(256);
    dim3 inputGradGrid((INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS + inputGradBlock.x - 1) / inputGradBlock.x);
    
    convolutionBackpropInputKernel<<<inputGradGrid, inputGradBlock>>>(
        d_conv_gradient, d_filters, d_input_gradient,
        OUTPUT_HEIGHT, OUTPUT_WIDTH, NUM_FILTERS,
        FILTER_SIZE, FILTER_SIZE, INPUT_CHANNELS,
        INPUT_HEIGHT, INPUT_WIDTH,
        PADDING, PADDING,
        STRIDE, STRIDE
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Print summary of dimensions and operations
    printf("\nCNN Architecture Summary:\n");
    printf("Input: %d x %d x %d\n", INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS);
    printf("Convolution: %d filters of size %d x %d, stride %d, padding %d\n", 
           NUM_FILTERS, FILTER_SIZE, FILTER_SIZE, STRIDE, PADDING);
    printf("After Convolution: %d x %d x %d\n", OUTPUT_HEIGHT, OUTPUT_WIDTH, NUM_FILTERS);
    printf("After ReLU: %d x %d x %d\n", OUTPUT_HEIGHT, OUTPUT_WIDTH, NUM_FILTERS);
    printf("Pooling: size %d x %d, stride %d\n", POOL_SIZE, POOL_SIZE, POOL_STRIDE);
    printf("After Pooling: %d x %d x %d\n", POOL_OUTPUT_HEIGHT, POOL_OUTPUT_WIDTH, NUM_FILTERS);
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_filters);
    cudaFree(d_bias);
    cudaFree(d_conv_output);
    cudaFree(d_relu_output);
    cudaFree(d_pool_output);
    cudaFree(d_pool_indices);
    cudaFree(d_im2col_data);
    cudaFree(d_pool_gradient);
    cudaFree(d_relu_gradient);
    cudaFree(d_conv_gradient);
    cudaFree(d_filter_gradient);
    cudaFree(d_input_gradient);
    
    // Free host memory
    free(h_input);
    free(h_filters);
    free(h_bias);
    free(h_pool_gradient);
    
    printf("\nCNN implementation completed successfully!\n");
    
    return 0;
}
