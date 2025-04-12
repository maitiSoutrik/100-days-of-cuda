#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath> // For fabs

// Error checking macro
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t error = call;                                                \
        if (error != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(error));                                  \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

// --- Shared Memory Reduction Kernel (from Day 4, renamed) ---

__global__ void reduceSharedMemKernel(const float *input, float *output, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x; // Stride of 2 elements per thread
    unsigned int gridSize = blockDim.x * 2 * gridDim.x;

    float mySum = 0.0f;

    // Each thread sums multiple elements if n > gridSize
    while (i < n) {
        mySum += input[i];
        // Handle second element if stride allows
        if (i + blockDim.x < n) {
            mySum += input[i + blockDim.x];
        }
        i += gridSize;
    }

    // Load partial sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// --- Warp Shuffle Reduction Kernel (Day 33) ---

// Helper function to reduce values within a warp using __shfl_down_sync
// Note: Requires CUDA 9.0+ and Compute Capability 3.0+ (__shfl_sync requires 7.0+)
// Jetson Nano is sm_53, so __shfl_down_sync is fine.
__device__ __forceinline__ float warpReduceSum(float val) {
    // Iterate log2(warpSize) times
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        // Shuffle value from thread `offset` lanes down and add to current thread's value
        // `0xFFFFFFFF` mask means all threads participate.
        // `warpSize` ensures we shuffle within the current warp.
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val; // Lane 0 holds the final sum for the warp
}

__global__ void reduceWarpShflKernel(const float *input, float *output, int n) {
    // Shared memory to store partial sums from each warp
    // Size needs to be blockDim.x / warpSize
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int laneId = tid % warpSize;       // Lane index within the warp (0-31)
    unsigned int warpId = tid / warpSize;       // Warp index within the block
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x; // Stride of 2 elements per thread
    unsigned int gridSize = blockDim.x * 2 * gridDim.x;

    float mySum = 0.0f;

    // Each thread sums multiple elements if n > gridSize
    // This part remains the same as the shared memory version
    while (i < n) {
        mySum += input[i];
        // Handle second element if stride allows
        if (i + blockDim.x < n) {
            mySum += input[i + blockDim.x];
        }
        i += gridSize;
    }

    // Perform warp-level reduction using shuffle
    mySum = warpReduceSum(mySum);

    // Lane 0 of each warp writes its partial sum to shared memory
    if (laneId == 0) {
        sdata[warpId] = mySum;
    }

    // Synchronize threads within the block to ensure shared memory is updated
    __syncthreads();

    // The first warp (specifically, thread 0 if blockDim.x >= warpSize)
    // performs the final reduction on the partial sums stored in shared memory.
    // We only need to reduce `gridDim.x / warpSize` elements now.
    if (tid < warpSize) { // Only need the first warp for this final reduction step
       // Load the partial sums from shared memory into the registers of the first warp
       // Need to ensure we don't read out of bounds if blockDim < warpSize
       int numWarpsInBlock = (blockDim.x + warpSize - 1) / warpSize;
       float warp_sum = (tid < numWarpsInBlock) ? sdata[tid] : 0.0f;

       // Reduce the sums within the first warp
       warp_sum = warpReduceSum(warp_sum);

       // Thread 0 (lane 0 of the first warp) writes the final block sum
       if (laneId == 0) {
           output[blockIdx.x] = warp_sum;
       }
    }
}


// CPU implementation of sum for verification
float sumArrayCPU(const float *arr, int n) {
    double sum = 0.0; // Use double for precision on CPU
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return (float)sum;
}

// Helper function to run and time a reduction kernel
float runReductionGPU(const char* kernel_name,
                      void (*kernelFunc)(const float*, float*, int),
                      float *h_input, int n, int blockSize, int numBlocks)
{
    float *d_input, *d_output;
    float h_sum = 0.0f;
    cudaEvent_t start, stop;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_output, numBlocks * sizeof(float)));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate host memory for output
    float *h_output = (float *)malloc(numBlocks * sizeof(float));
    if (!h_output) {
        fprintf(stderr, "Failed to allocate host memory for output\n");
        exit(EXIT_FAILURE);
    }

    // Create CUDA events for timing
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Determine shared memory size
    // Shared memory version needs blockDim.x floats
    // Warp shuffle version needs blockDim.x / 32 floats (rounded up if not multiple)
    // NOTE: Using literal 32 here as warpSize is not available in host code
    size_t sharedMemSize = (kernelFunc == reduceSharedMemKernel) ?
                           blockSize * sizeof(float) :
                           ((blockSize + 32 - 1) / 32) * sizeof(float);


    // Launch kernel and record time
    CUDA_CHECK(cudaEventRecord(start));
    kernelFunc<<<numBlocks, blockSize, sharedMemSize>>>(d_input, d_output, n);
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop)); // Wait for kernel completion

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy output data back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));

    // Sum the partial sums on CPU
    // Use double for host-side summation for better precision
    double host_sum_double = 0.0;
    for (int i = 0; i < numBlocks; i++) {
        host_sum_double += h_output[i];
    }
    h_sum = (float)host_sum_double;

    // Print timing
    printf("GPU Kernel (%s): Sum = %.1f, Time: %.6f ms\n", kernel_name, h_sum, milliseconds);

    // Free memory
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_output);

    return h_sum;
}


int main(int argc, char *argv[]) {
    // Default array size (e.g., 2^24 = 16,777,216)
    int n = 1 << 24;

    if (argc > 1) {
        n = atoi(argv[1]);
        if (n <= 0) {
            fprintf(stderr, "Invalid array size: %s\n", argv[1]);
            return 1;
        }
    }

    printf("Performing parallel reduction sum on %d float elements\n", n);

    // Allocate and initialize host array
    float *h_input = (float *)malloc(n * sizeof(float));
     if (!h_input) {
        fprintf(stderr, "Failed to allocate host memory for input\n");
        return 1;
    }
    // Initialize with a value that makes verification easy but avoids simple cancellation
    for (int i = 0; i < n; i++) {
        // Use a simple pattern for easy verification if needed, but avoid exactly 1.0
        // h_input[i] = 1.0f;
        h_input[i] = (float)(i % 100) * 0.1f + 0.5f; // Small varying values
    }

    // --- CPU Reduction ---
    printf("\n--- CPU Computation ---\n");
    auto cpu_start = std::chrono::high_resolution_clock::now();
    float cpu_sum = sumArrayCPU(h_input, n);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time_ms = cpu_end - cpu_start;
    printf("CPU Sum: %.1f, Time: %.6f ms\n", cpu_sum, cpu_time_ms.count());


    // --- GPU Reductions ---
    printf("\n--- GPU Computations ---\n");

    // Determine block size and number of blocks
    // Block size should ideally be a multiple of warpSize (32)
    int blockSize = 256; // Common choice, multiple of 32
    // NOTE: Using literal 32 here as warpSize is not available in host code
    if (blockSize % 32 != 0 && blockSize > 0) {
       // This should not happen with blockSize=256, but good practice to check
       fprintf(stderr, "Warning: Block size %d not a multiple of warpSize (32). Consider adjusting.\n", blockSize);
    }

    // Calculate numBlocks needed. The output array stores one sum per block.
    // Each thread processes elements with a stride of blockDim.x * 2 initially
    // Then reduction happens per block.
    // Effective elements processed per block launch = blockDim.x * 2
    int elements_per_block_pass = blockSize * 2;
    // Number of blocks needed to cover all elements 'n'
    int numBlocks = (n + elements_per_block_pass - 1) / elements_per_block_pass;
     // Ensure numBlocks is at least 1 if n > 0
    if (numBlocks == 0 && n > 0) numBlocks = 1;
    printf("GPU Config: BlockSize = %d, NumBlocks = %d\n", blockSize, numBlocks);


    // Run Shared Memory version
    float gpu_sum_shared = runReductionGPU("Shared Memory", reduceSharedMemKernel, h_input, n, blockSize, numBlocks);

    // Run Warp Shuffle version
    float gpu_sum_warp = runReductionGPU("Warp Shuffle", reduceWarpShflKernel, h_input, n, blockSize, numBlocks);


    // --- Verification ---
    printf("\n--- Verification ---\n");
    // Tolerance needs to account for potential floating-point differences
    // especially with large sums and different reduction orders.
    float tolerance = fabsf(cpu_sum) * 1e-5; // Relative tolerance
    // Add a small absolute tolerance for cases where cpu_sum is near zero
    tolerance = fmaxf(tolerance, (float)n * 1e-6); // Absolute tolerance component

    bool shared_passed = fabsf(cpu_sum - gpu_sum_shared) <= tolerance;
    bool warp_passed = fabsf(cpu_sum - gpu_sum_warp) <= tolerance;

    printf("Shared Memory GPU vs CPU: %s (CPU=%.1f, GPU=%.1f, Diff=%e, Tol=%e)\n",
           shared_passed ? "PASSED" : "FAILED", cpu_sum, gpu_sum_shared, fabsf(cpu_sum - gpu_sum_shared), tolerance);
    printf("Warp Shuffle GPU vs CPU: %s (CPU=%.1f, GPU=%.1f, Diff=%e, Tol=%e)\n",
           warp_passed ? "PASSED" : "FAILED", cpu_sum, gpu_sum_warp, fabsf(cpu_sum - gpu_sum_warp), tolerance);


    // Free host memory
    free(h_input);

    return (shared_passed && warp_passed) ? 0 : 1; // Return error code if verification fails
}
