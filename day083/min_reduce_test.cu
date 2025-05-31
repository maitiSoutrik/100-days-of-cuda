#include "min_reduce.cuh"
#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <cfloat> // For FLT_MAX

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
        for (size_t s : shape) total_input_elements *= s;

        if (dim_to_reduce < 0 || static_cast<size_t>(dim_to_reduce) >= shape.size()) {
            GTEST_SKIP() << "Invalid dimension to reduce: " << dim_to_reduce << " for shape size " << shape.size();
            return;
        }
        
        if (shape[dim_to_reduce] == 0) {
            // The main code has a check for this and returns.
            // For GTest, we can assert specific behavior or skip.
            // Let's assume the CUDA function handles this by not crashing and possibly producing an empty/default output.
            // Here, we'll skip direct testing of this case as the function is expected to return early.
            // Or, we can test that it *does* return early without error.
            // For now, let's verify it doesn't crash.
             std::vector<float> h_input(total_input_elements > 0 ? total_input_elements : 1, 1.0f); // Dummy input
             std::vector<size_t> output_shape_vec;
             size_t total_output_elements = 0; // Expect 0 output elements
             std::vector<float> h_output_gpu(1); // Dummy output

             float *d_input, *d_output;
             CHECK_CUDA_ERROR_GTEST(cudaMalloc(&d_input, (total_input_elements > 0 ? total_input_elements : 1) * sizeof(float)));
             CHECK_CUDA_ERROR_GTEST(cudaMalloc(&d_output, 1 * sizeof(float))); // Minimal allocation
             CHECK_CUDA_ERROR_GTEST(cudaMemcpy(d_input, h_input.data(), (total_input_elements > 0 ? total_input_elements : 1) * sizeof(float), cudaMemcpyHostToDevice));

             min_reduction_dimension_cuda(d_input, dim_to_reduce, d_output, shape.data(), shape.size());
             CHECK_CUDA_ERROR_GTEST(cudaDeviceSynchronize()); // Check for launch errors

             CHECK_CUDA_ERROR_GTEST(cudaFree(d_input));
             CHECK_CUDA_ERROR_GTEST(cudaFree(d_output));
             SUCCEED() << "Test with zero-sized reduction dimension ran without CUDA errors (expected early exit).";
            return;
        }


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
        
        if (total_output_elements == 0 && total_input_elements > 0) {
             // This means a non-reduced dimension was zero. Output is legitimately empty.
             // The CUDA kernel should handle this by its output_size check or numBlocks being 0.
             std::vector<float> h_output_gpu(1); // Dummy, won't be written to if output_size is 0
             float *d_input, *d_output;
             CHECK_CUDA_ERROR_GTEST(cudaMalloc(&d_input, total_input_elements * sizeof(float)));
             CHECK_CUDA_ERROR_GTEST(cudaMalloc(&d_output, 1 * sizeof(float))); // Minimal allocation
             CHECK_CUDA_ERROR_GTEST(cudaMemcpy(d_input, h_input.data(), total_input_elements * sizeof(float), cudaMemcpyHostToDevice));
             
             min_reduction_dimension_cuda(d_input, dim_to_reduce, d_output, shape.data(), shape.size());
             CHECK_CUDA_ERROR_GTEST(cudaDeviceSynchronize());

             CHECK_CUDA_ERROR_GTEST(cudaFree(d_input));
             CHECK_CUDA_ERROR_GTEST(cudaFree(d_output));
             SUCCEED() << "Test with zero-sized output (due to non-reduced dim being 0) ran without CUDA errors.";
             return;
        }


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
