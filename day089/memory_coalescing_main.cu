#include "memory_coalescing.cuh"
#include <vector>
#include <numeric>
#include <algorithm>
#include <iomanip> // For std::fixed and std::setprecision

// Helper function to print a small part of an array
void print_sample(const std::string& name, const float* arr, int n, int count = 16) {
    std::cout << name << " (first " << std::min(n, count) << " elements): [";
    for (int i = 0; i < std::min(n, count); ++i) {
        std::cout << arr[i] << (i == std::min(n, count) - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;
}

// Helper function to verify results
bool verify_results(const float* arr1, const float* arr2, int n) {
    const float epsilon = 1e-5f;
    for (int i = 0; i < n; ++i) {
        if (abs(arr1[i] - arr2[i]) > epsilon) {
            std::cerr << "Verification FAILED at index " << i << ": arr1 = " << arr1[i] << ", arr2 = " << arr2[i] << std::endl;
            return false;
        }
    }
    std::cout << "Verification PASSED!" << std::endl;
    return true;
}

int main() {
    const int n = 1024 * 1024 * 16; // 16M elements, ~64MB for a float array
    const float scalar = 2.5f;
    const int stride_factor = 32; // For uncoalesced access. Should be >= warpSize for good effect.

    std::cout << "Number of elements: " << n << std::endl;
    std::cout << "Scalar: " << scalar << std::endl;
    std::cout << "Stride factor for uncoalesced kernel: " << stride_factor << std::endl;

    // Allocate host memory
    std::vector<float> h_input(n);
    std::vector<float> h_output_coalesced(n);
    std::vector<float> h_output_uncoalesced(n);
    std::vector<float> h_expected_output(n); // For coalesced output
    std::vector<float> h_expected_output_permuted_for_uncoalesced(n); // For uncoalesced output

    // Initialize input data and expected output for coalesced kernel
    for (int i = 0; i < n; ++i) {
        h_input[i] = static_cast<float>(i % 100); // Simple pattern
        h_expected_output[i] = h_input[i] * scalar;
    }
    
    // Initialize expected output for uncoalesced kernel (permuted)
    int elements_per_stride_group = n / stride_factor;
    if (elements_per_stride_group == 0 && n > 0) elements_per_stride_group = 1;

    for(int global_thread_idx = 0; global_thread_idx < n; ++global_thread_idx) {
        if (elements_per_stride_group > 0) {
            int group_idx = global_thread_idx / elements_per_stride_group;
            int idx_in_group = global_thread_idx % elements_per_stride_group;
            if (group_idx < stride_factor) {
                int uncoalesced_idx = idx_in_group * stride_factor + group_idx;
                if (uncoalesced_idx < n) {
                    h_expected_output_permuted_for_uncoalesced[uncoalesced_idx] = h_input[uncoalesced_idx] * scalar;
                }
            }
        }
    }

    // Allocate device memory
    float *d_input, *d_output_coalesced, *d_output_uncoalesced;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_coalesced, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_uncoalesced, n * sizeof(float)));

    // Copy input data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel launch parameters
    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    // CUDA Events for timing
    cudaEvent_t start_event, stop_event;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_event));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_event));
    float milliseconds = 0;

    // --- Coalesced Kernel ---
    std::cout << "\n--- Running Coalesced Access Kernel ---" << std::endl;
    CHECK_CUDA_ERROR(cudaEventRecord(start_event));
    coalesced_access_kernel<<<blocks_per_grid, threads_per_block>>>(d_input, d_output_coalesced, n, scalar);
    CHECK_CUDA_ERROR(cudaEventRecord(stop_event));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_event));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
    std::cout << "Coalesced Kernel Execution Time: " << std::fixed << std::setprecision(3) << milliseconds << " ms" << std::endl;
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_coalesced.data(), d_output_coalesced, n * sizeof(float), cudaMemcpyDeviceToHost));
    print_sample("Coalesced Output", h_output_coalesced.data(), n);
    verify_results(h_output_coalesced.data(), h_expected_output.data(), n);

    // --- Uncoalesced Kernel ---
    std::cout << "\n--- Running Uncoalesced Access Kernel ---" << std::endl;
    CHECK_CUDA_ERROR(cudaMemset(d_output_uncoalesced, 0, n * sizeof(float))); 
    CHECK_CUDA_ERROR(cudaEventRecord(start_event));
    uncoalesced_access_kernel<<<blocks_per_grid, threads_per_block>>>(d_input, d_output_uncoalesced, n, scalar, stride_factor);
    CHECK_CUDA_ERROR(cudaEventRecord(stop_event));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_event));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
    std::cout << "Uncoalesced Kernel Execution Time: " << std::fixed << std::setprecision(3) << milliseconds << " ms" << std::endl;
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_uncoalesced.data(), d_output_uncoalesced, n * sizeof(float), cudaMemcpyDeviceToHost));
    print_sample("Uncoalesced Output", h_output_uncoalesced.data(), n);
    verify_results(h_output_uncoalesced.data(), h_expected_output_permuted_for_uncoalesced.data(), n);


    // Cleanup
    CHECK_CUDA_ERROR(cudaEventDestroy(start_event));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_event));
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output_coalesced));
    CHECK_CUDA_ERROR(cudaFree(d_output_uncoalesced));

    return 0;
}

