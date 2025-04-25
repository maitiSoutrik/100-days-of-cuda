#include <cuda_runtime.h> // Core CUDA runtime functions and types
#include <iostream>
#include <vector>
#include <cstdio> // For printf in kernels

// CUDA runtime error checking macro
#define CHECK_CUDA_ERROR(val) checkCuda((val), #val, __FILE__, __LINE__)
void checkCuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), cudaGetErrorName(result), func);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

// Child kernel: Does a simple task, like writing its unique ID
__global__ void child_kernel(int* output_data, int parent_tid, int child_tid) {
    // Calculate a unique index for this child invocation
    // Note: blockDim.x for the child is 1 in this example
    int idx = parent_tid; // Each parent thread launches one child
    output_data[idx] = parent_tid * 1000 + child_tid; // Store a value indicating parent and child
    // printf("Child kernel from parent %d, child %d writing to index %d\n", parent_tid, child_tid, idx);
}

// Parent kernel: Launches the child kernel
__global__ void parent_kernel(int* output_data, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        // Launch the child kernel dynamically
        // Each thread in the parent grid launches one child kernel thread
        // The child kernel grid is 1x1x1, and the block is 1x1x1
        child_kernel<<<1, 1>>>(output_data, tid, 0); // Pass parent tid and a child id (0 here)

        // It's important to synchronize within the parent kernel if the parent
        // needs to read results produced by its own child kernels *before* the
        // parent kernel exits. cudaDeviceSynchronize() inside a kernel waits for
        // **all** child launches *initiated by that specific thread* to complete.
        // If other threads in the parent might write to the same output location
        // or if subsequent parent logic depends on child results, synchronization is crucial.
        // In this simple example, each parent writes to a unique location, and there's
        // no immediate dependency within the parent kernel, but it's good practice.
         cudaError_t syncResult = cudaDeviceSynchronize();
         if (syncResult != cudaSuccess) {
             // Handle error, maybe write an error code to a specific memory location
             // printf inside kernel is generally discouraged for performance reasons but okay for debugging
             printf("Parent tid %d: cudaDeviceSynchronize ERROR: %s\n", tid, cudaGetErrorString(syncResult));
         }
    }
}

int main() {
    int size = 256; // Number of parent threads and output elements
    std::cout << "Running Dynamic Parallelism Example with size: " << size << std::endl;

    // Allocate host memory
    std::vector<int> h_output_data(size, -1); // Initialize with -1

    // Allocate device memory
    int* d_output_data = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_data, size * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(d_output_data, 0, size * sizeof(int))); // Initialize device memory

    // Kernel launch configuration
    int threads_per_block = 128;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    std::cout << "Launching parent kernel with " << blocks_per_grid << " blocks and "
              << threads_per_block << " threads per block." << std::endl;

    // --- Timing Setup ---
    cudaEvent_t start_event, stop_event;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_event));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_event));

    // Record start event
    CHECK_CUDA_ERROR(cudaEventRecord(start_event, 0)); // 0 is the default stream

    // --- Launch the parent kernel ---
    parent_kernel<<<blocks_per_grid, threads_per_block>>>(d_output_data, size);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors

    // --- Timing End & Measurement ---
    // Record stop event
    CHECK_CUDA_ERROR(cudaEventRecord(stop_event, 0));

    // Synchronize the host thread with the stop event specifically
    // This ensures the kernel (and children) is finished before calculating time
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_event));
    std::cout << "Parent kernel execution completed." << std::endl;

    // Calculate elapsed time
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
    std::cout << "Kernel Execution Time (including children): " << milliseconds << " ms" << std::endl;

    // Destroy events
    CHECK_CUDA_ERROR(cudaEventDestroy(start_event));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_event));
    // --- Timing End ---


    // Copy results back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_data.data(), d_output_data, size * sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "Results copied back to host." << std::endl;

    // Verify results
    bool success = true;
    int errors = 0;
    for (int i = 0; i < size; ++i) {
        int expected_value = i * 1000 + 0; // Based on child_kernel logic
        if (h_output_data[i] != expected_value) {
            if (errors < 10) { // Print first few errors
                 printf("Verification failed at index %d: Expected %d, Got %d\n", i, expected_value, h_output_data[i]);
            }
            success = false;
            errors++;
        }
    }

    if (success) {
        std::cout << "Verification successful!" << std::endl;
        // Optionally print some results
        std::cout << "First few output values: ";
        for(int i = 0; i < std::min(size, 10); ++i) {
            std::cout << h_output_data[i] << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "Verification failed with " << errors << " errors." << std::endl;
    }

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_output_data));
    std::cout << "Device memory freed." << std::endl;

    // Reset device (optional, good practice)
    CHECK_CUDA_ERROR(cudaDeviceReset());

    return success ? 0 : 1;
}
