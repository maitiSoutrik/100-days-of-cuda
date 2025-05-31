#include "min_reduce.cuh"
#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <cfloat> // For FLT_MAX
#include <unistd.h> // For dup, dup2, close, STDERR_FILENO
#include <fcntl.h>  // For open, O_WRONLY

// Error checking macro
#define CHECK_CUDA_ERROR_GTEST(err) \
    ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err)

// CPU implementation for verification (can be shared or duplicated from main)
void min_reduction_dimension_cpu_gtest(
    const float* input,
    int dim_to_reduce,
    float* output_cpu,
    const std::vector<size_t>& shape_vec,
    const std::vector<size_t>& output_shape_vec // Expected output shape
) {
    size_t ndim = shape_vec.size();
    size_t before_size = 1;
    for (int i = 0; i < dim_to_reduce; ++i) {
        before_size *= shape_vec[i];
    }
    size_t dim_size = shape_vec[dim_to_reduce];
    size_t after_size = 1;
    for (size_t i = dim_to_reduce + 1; i < ndim; ++i) {
        after_size *= shape_vec[i];
    }

    size_t output_idx_counter = 0;
    // Ensure output_cpu is sized correctly based on output_shape_vec
    size_t expected_output_elements = 1;
    if (output_shape_vec.empty() && before_size * after_size == 1 && dim_size > 0) { // scalar output
         expected_output_elements = 1;
    } else {
        for(size_t s : output_shape_vec) expected_output_elements *= s;
    }


    for (size_t b = 0; b < before_size; ++b) {
        for (size_t a = 0; a < after_size; ++a) {
            float min_val = FLT_MAX;
            if (dim_size > 0) { // Only iterate if the dimension to reduce is not empty
                for (size_t d = 0; d < dim_size; ++d) {
                    size_t input_idx = (b * dim_size + d) * after_size + a;
                    min_val = std::min(min_val, input[input_idx]);
                }
            } else { // If dim_size is 0, behavior might be undefined or specific (e.g. FLT_MAX or 0)
                      // For this test, let's assume if dim_size is 0, the output for that element is FLT_MAX
                      // This case should ideally be prevented by checks before calling.
                min_val = FLT_MAX; // Or some other indicator of an empty reduction
            }
            if (output_idx_counter < expected_output_elements) {
                 output_cpu[output_idx_counter++] = min_val;
            } else {
                // This should not happen if logic is correct
                FAIL() << "Output index out of bounds in CPU reference calculation.";
            }
        }
    }
}


class MinReductionTest : public ::testing::TestWithParam<std::tuple<std::vector<size_t>, int>> {
protected:
    void runTest(const std::vector<size_t>& shape, int dim_to_reduce) {
        size_t total_input_elements = 1;
        bool any_input_shape_dim_is_zero = false;
        for (size_t s : shape) {
            if (s == 0) any_input_shape_dim_is_zero = true;
            total_input_elements *= s;
        }

        if (dim_to_reduce < 0 || static_cast<size_t>(dim_to_reduce) >= shape.size()) {
            GTEST_SKIP() << "Invalid dimension to reduce: " << dim_to_reduce << " for shape size " << shape.size();
            return;
        }
        
        // Case 1: The dimension to be reduced is itself of size 0.
        // Example: shape = {5, 0, 6}, dim_to_reduce = 1.
        // Expected: min_reduction_dimension_cuda prints "Error: Size of dimension to reduce..." and returns.
        if (shape[dim_to_reduce] == 0) {
            // Redirect stderr to /dev/null to suppress the expected error message from the function
            int stderr_original_fd = dup(STDERR_FILENO);
            ASSERT_NE(stderr_original_fd, -1) << "Failed to dup stderr";
            
            int dev_null_fd = open("/dev/null", O_WRONLY);
            ASSERT_NE(dev_null_fd, -1) << "Failed to open /dev/null";
            
            ASSERT_NE(dup2(dev_null_fd, STDERR_FILENO), -1) << "Failed to redirect stderr to /dev/null";
            close(dev_null_fd); // Close the descriptor for /dev/null as it's now duplicated to stderr

            float *d_input_dummy, *d_output_dummy;
            std::vector<float> h_dummy(1, 0.0f); 

            CHECK_CUDA_ERROR_GTEST(cudaMalloc(&d_input_dummy, 1 * sizeof(float)));
            CHECK_CUDA_ERROR_GTEST(cudaMalloc(&d_output_dummy, 1 * sizeof(float)));
            CHECK_CUDA_ERROR_GTEST(cudaMemcpy(d_input_dummy, h_dummy.data(), 1 * sizeof(float), cudaMemcpyHostToDevice));
            
            min_reduction_dimension_cuda(d_input_dummy, dim_to_reduce, d_output_dummy, shape.data(), shape.size());
            CHECK_CUDA_ERROR_GTEST(cudaDeviceSynchronize()); 

            CHECK_CUDA_ERROR_GTEST(cudaFree(d_input_dummy));
            CHECK_CUDA_ERROR_GTEST(cudaFree(d_output_dummy));

            // Restore stderr
            fflush(stderr); // Ensure any buffered messages to the redirected stderr are flushed
            ASSERT_NE(dup2(stderr_original_fd, STDERR_FILENO), -1) << "Failed to restore stderr";
            close(stderr_original_fd);
            
            SUCCEED() << "Test for reducing a zero-sized dimension: function correctly handled the case (stderr suppressed for test).";
            return;
        }

        // Case 2: An input dimension *not* being reduced is 0, leading to total_input_elements = 0.
        // Example: shape = {5, 0, 6}, dim_to_reduce = 0. Here total_input_elements is 0.
        // Expected: min_reduction_dimension_cuda should return early (e.g., due to output_size == 0)
        // without "Null pointer" errors, as we provide valid (dummy) pointers.
        if (total_input_elements == 0 && any_input_shape_dim_is_zero) {
            float *d_input_dummy, *d_output_dummy;
            std::vector<float> h_dummy(1, 0.0f);

            CHECK_CUDA_ERROR_GTEST(cudaMalloc(&d_input_dummy, 1 * sizeof(float)));
            CHECK_CUDA_ERROR_GTEST(cudaMalloc(&d_output_dummy, 1 * sizeof(float)));
            CHECK_CUDA_ERROR_GTEST(cudaMemcpy(d_input_dummy, h_dummy.data(), 1 * sizeof(float), cudaMemcpyHostToDevice));

            min_reduction_dimension_cuda(d_input_dummy, dim_to_reduce, d_output_dummy, shape.data(), shape.size());
            CHECK_CUDA_ERROR_GTEST(cudaDeviceSynchronize());

            CHECK_CUDA_ERROR_GTEST(cudaFree(d_input_dummy));
            CHECK_CUDA_ERROR_GTEST(cudaFree(d_output_dummy));
            SUCCEED() << "Test for zero total input elements (due to a non-reduced dim being 0): function expected to return early.";
            return;
        }
        
        // Default case: Valid inputs, proceed with full test
        ASSERT_GT(total_input_elements, 0) << "Test logic assumes non-zero total elements at this point.";

        std::vector<float> h_input(total_input_elements);
        std::mt19937 rng(12345); // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
        std::generate(h_input.begin(), h_input.end(), [&]() { return dist(rng); });

        std::vector<size_t> output_shape_vec;
        size_t total_output_elements = 1;
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i != static_cast<size_t>(dim_to_reduce)) {
                output_shape_vec.push_back(shape[i]);
                total_output_elements *= shape[i];
            }
        }
        if (output_shape_vec.empty() && total_input_elements > 0) { // Reduction to scalar
            output_shape_vec.push_back(1);
            total_output_elements = 1;
        }
        // If total_input_elements > 0, then total_output_elements must also be > 0 (or 1 for scalar result)
        // The block for `total_output_elements == 0 && total_input_elements > 0` was determined to be unreachable.
        ASSERT_GT(total_output_elements, 0) << "Output elements should be > 0 if input elements > 0.";

        std::vector<float> h_output_gpu(total_output_elements);
        std::vector<float> h_output_cpu(total_output_elements);

        float *d_input, *d_output;
        CHECK_CUDA_ERROR_GTEST(cudaMalloc(&d_input, total_input_elements * sizeof(float)));
        CHECK_CUDA_ERROR_GTEST(cudaMalloc(&d_output, total_output_elements * sizeof(float)));

        CHECK_CUDA_ERROR_GTEST(cudaMemcpy(d_input, h_input.data(), total_input_elements * sizeof(float), cudaMemcpyHostToDevice));

        min_reduction_dimension_cuda(d_input, dim_to_reduce, d_output, shape.data(), shape.size());
        CHECK_CUDA_ERROR_GTEST(cudaDeviceSynchronize());

        CHECK_CUDA_ERROR_GTEST(cudaMemcpy(h_output_gpu.data(), d_output, total_output_elements * sizeof(float), cudaMemcpyDeviceToHost));

        min_reduction_dimension_cpu_gtest(h_input.data(), dim_to_reduce, h_output_cpu.data(), shape, output_shape_vec);

        for (size_t i = 0; i < total_output_elements; ++i) {
            ASSERT_FLOAT_EQ(h_output_cpu[i], h_output_gpu[i]) << "Mismatch at index " << i;
        }

        CHECK_CUDA_ERROR_GTEST(cudaFree(d_input));
        CHECK_CUDA_ERROR_GTEST(cudaFree(d_output));
    }
};

TEST_P(MinReductionTest, HandlesVariousShapesAndDims) {
    const auto& params = GetParam();
    runTest(std::get<0>(params), std::get<1>(params));
}

INSTANTIATE_TEST_SUITE_P(
    MinReductionTests,
    MinReductionTest,
    ::testing::Values(
        // Basic 3D tests
        std::make_tuple(std::vector<size_t>{2, 3, 4}, 0),
        std::make_tuple(std::vector<size_t>{2, 3, 4}, 1),
        std::make_tuple(std::vector<size_t>{2, 3, 4}, 2),
        // Larger 2D tests
        std::make_tuple(std::vector<size_t>{64, 128}, 0),
        std::make_tuple(std::vector<size_t>{128, 64}, 1),
        // 1D array (reduction to scalar)
        std::make_tuple(std::vector<size_t>{100}, 0),
        // Dimension of size 1
        std::make_tuple(std::vector<size_t>{10, 1, 10}, 0),
        std::make_tuple(std::vector<size_t>{10, 1, 10}, 1),
        std::make_tuple(std::vector<size_t>{10, 1, 10}, 2),
        // 4D tensor
        std::make_tuple(std::vector<size_t>{2, 2, 3, 2}, 0),
        std::make_tuple(std::vector<size_t>{2, 2, 3, 2}, 1),
        std::make_tuple(std::vector<size_t>{2, 2, 3, 2}, 2),
        std::make_tuple(std::vector<size_t>{2, 2, 3, 2}, 3),
        // Larger, more complex shapes
        std::make_tuple(std::vector<size_t>{16, 32, 8}, 1),
        std::make_tuple(std::vector<size_t>{7, 11, 13}, 0),
        std::make_tuple(std::vector<size_t>{7, 11, 13}, 1),
        std::make_tuple(std::vector<size_t>{7, 11, 13}, 2),
        // Test with a zero dimension (not reduced) - leads to zero output elements
        std::make_tuple(std::vector<size_t>{5, 0, 6}, 0), // Output shape {0,6}
        std::make_tuple(std::vector<size_t>{5, 0, 6}, 2), // Output shape {5,0}
        // Test with a zero dimension (reduced) - handled by shape[dim_to_reduce] == 0 check
        std::make_tuple(std::vector<size_t>{5, 0, 6}, 1) 
    )
);

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
