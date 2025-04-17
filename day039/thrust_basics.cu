#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>

#include <iostream>
#include <vector>
#include <numeric> // For std::iota, std::accumulate
#include <cstdlib> // For rand()
#include <ctime>   // For clock()
#include <cmath>   // For fabs

// Macro for checking CUDA errors
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Functor for generating random numbers
struct RandomGenerator {
    __host__ __device__
    int operator()(unsigned int idx) const {
        // Simple linear congruential generator (for demonstration)
        // Note: Using device-side rand() directly within a functor can be tricky.
        // For better randomness, consider thrust::random or cuRAND.
        // This simple generator is just for filling the vector quickly.
        unsigned long long seed = idx + 1;
        seed = (1103515245 * seed + 12345) & 0x7FFFFFFF; // Keep it positive
        return static_cast<int>(seed % 100); // Generate numbers between 0 and 99
    }
};

// Helper function to print vectors (useful for debugging small examples)
template <typename T>
void print_vector(const std::string& name, const T& vec, size_t n = 10) {
    std::cout << name << " (first " << std::min(n, vec.size()) << " elements): ";
    for (size_t i = 0; i < std::min(n, vec.size()); ++i) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

int main(void) {
    size_t N = 1 << 20; // 1 Million elements

    // === 1. Thrust Reduction Example ===
    std::cout << "--- Thrust Reduction Example ---" << std::endl;
    thrust::device_vector<int> d_vec_reduce(N);

    // Initialize vector with random numbers using Thrust's sequence and transform (or generate)
    // Using sequence + transform often gives more control than a simple generator functor
    thrust::sequence(d_vec_reduce.begin(), d_vec_reduce.end()); // Fill with 0, 1, 2, ...
    thrust::transform(d_vec_reduce.begin(), d_vec_reduce.end(), d_vec_reduce.begin(),
                      [] __device__ (int x) { return (x % 100); }); // Transform to pseudo-random 0-99

    // Time the reduction
    cudaEvent_t start_reduce, stop_reduce;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_reduce));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_reduce));

    CHECK_CUDA_ERROR(cudaEventRecord(start_reduce));
    int sum = thrust::reduce(d_vec_reduce.begin(), d_vec_reduce.end(), 0, thrust::plus<int>());
    CHECK_CUDA_ERROR(cudaEventRecord(stop_reduce));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_reduce));

    float milliseconds_reduce = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds_reduce, start_reduce, stop_reduce));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_reduce));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_reduce));

    std::cout << "Thrust reduction sum: " << sum << std::endl;
    std::cout << "Thrust reduction time: " << milliseconds_reduce << " ms" << std::endl;

    // Verification (CPU reduction) - Only for smaller N due to performance
    if (N <= (1 << 16)) { // Only verify for smaller sizes
        thrust::host_vector<int> h_vec_reduce = d_vec_reduce; // Copy back to host
        long long cpu_sum = 0;
        for(size_t i = 0; i < h_vec_reduce.size(); ++i) {
            cpu_sum += h_vec_reduce[i];
        }
        std::cout << "CPU reduction sum (verification): " << cpu_sum << std::endl;
        if (cpu_sum != sum) {
            std::cerr << "Verification FAILED for reduction!" << std::endl;
        } else {
            std::cout << "Verification PASSED for reduction." << std::endl;
        }
    } else {
         std::cout << "Skipping CPU verification for reduction due to large N." << std::endl;
    }
    std::cout << std::endl;


    // === 2. Thrust Inclusive Scan Example ===
    std::cout << "--- Thrust Inclusive Scan Example ---" << std::endl;
    thrust::device_vector<int> d_vec_scan_in(N);
    thrust::device_vector<int> d_vec_scan_out(N);

    // Initialize with 1s for a simple scan example (scan should produce 1, 2, 3, ...)
    thrust::fill(d_vec_scan_in.begin(), d_vec_scan_in.end(), 1);

    // Time the scan
    cudaEvent_t start_scan, stop_scan;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_scan));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_scan));

    CHECK_CUDA_ERROR(cudaEventRecord(start_scan));
    thrust::inclusive_scan(d_vec_scan_in.begin(), d_vec_scan_in.end(), d_vec_scan_out.begin());
    CHECK_CUDA_ERROR(cudaEventRecord(stop_scan));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_scan));

    float milliseconds_scan = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds_scan, start_scan, stop_scan));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_scan));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_scan));

    std::cout << "Thrust inclusive_scan time: " << milliseconds_scan << " ms" << std::endl;

    // Verification (check last element)
    int last_element = d_vec_scan_out[N - 1]; // Access last element (potentially slow, ok for verification)
    std::cout << "Last element of scan result: " << last_element << std::endl;
    if (last_element != static_cast<int>(N)) {
         std::cerr << "Verification FAILED for inclusive scan (expected last element " << N << ")" << std::endl;
    } else {
         std::cout << "Verification PASSED for inclusive scan (last element)." << std::endl;
    }
    // Optional: print first few elements for small N
    if (N <= 16) {
        thrust::host_vector<int> h_vec_scan_out = d_vec_scan_out;
        print_vector("Scan Result (GPU)", h_vec_scan_out, N);
    }
    std::cout << std::endl;


    // === 3. Thrust Sort Example ===
    std::cout << "--- Thrust Sort Example ---" << std::endl;
    thrust::device_vector<int> d_vec_sort(N);

    // Initialize with reverse sorted order (worst case for some sorts)
    // Use thrust::sequence and a simple transform
    thrust::sequence(d_vec_sort.begin(), d_vec_sort.end(), 0); // 0, 1, 2, ...
    thrust::transform(d_vec_sort.begin(), d_vec_sort.end(), d_vec_sort.begin(),
                      [N] __device__ (int x) { return N - 1 - x; }); // N-1, N-2, ..., 0


    // Time the sort
    cudaEvent_t start_sort, stop_sort;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_sort));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_sort));

    CHECK_CUDA_ERROR(cudaEventRecord(start_sort));
    thrust::sort(d_vec_sort.begin(), d_vec_sort.end());
    CHECK_CUDA_ERROR(cudaEventRecord(stop_sort));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_sort));

    float milliseconds_sort = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds_sort, start_sort, stop_sort));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_sort));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_sort));

    std::cout << "Thrust sort time: " << milliseconds_sort << " ms" << std::endl;

    // Verification (check if sorted)
    thrust::device_vector<int> d_vec_expected(N);
    thrust::sequence(d_vec_expected.begin(), d_vec_expected.end(), 0); // Expected: 0, 1, 2, ...

    // Use thrust::equal for efficient comparison on the device
    bool is_sorted = thrust::equal(d_vec_sort.begin(), d_vec_sort.end(), d_vec_expected.begin());

    if (!is_sorted) {
         std::cerr << "Verification FAILED for sort!" << std::endl;
         // For debugging small cases:
         if (N <= 16) {
             thrust::host_vector<int> h_vec_sort = d_vec_sort;
             print_vector("Sorted Vec (GPU)", h_vec_sort, N);
             thrust::host_vector<int> h_vec_expected = d_vec_expected;
             print_vector("Expected Vec", h_vec_expected, N);
         }
    } else {
         std::cout << "Verification PASSED for sort." << std::endl;
    }
    std::cout << std::endl;

    std::cout << "All Thrust examples completed." << std::endl;

    return 0;
}
