#include "min_reduce.cuh"
#include <iostream>
#include <vector>
#include <numeric>   // For std::iota
#include <algorithm> // For std::generate, std::min_element
#include <random>    // For std::mt19937, std::uniform_real_distribution
#include <iomanip>   // For std::fixed, std::setprecision
#include <chrono>    // For timing
#include <cfloat>    // For FLT_MAX

// Error checking macro
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

// Helper function to print a flattened array as a multi-dimensional tensor
void print_tensor(const float* tensor, const std::vector<size_t>& shape, const std::string& name) {
    std::cout << name << " (Shape: ";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << (i == shape.size() - 1 ? "" : "x");
    }
    std::cout << "):" << std::endl;

    if (shape.empty()) {
        std::cout << "[]" << std::endl;
        return;
    }

    size_t total_elements = 1;
    for (size_t dim_size : shape) {
        total_elements *= dim_size;
    }
    if (total_elements == 0) {
        std::cout << "[] (empty tensor)" << std::endl;
        return;
    }
    if (total_elements > 100) { // Limit printing for very large tensors
        std::cout << "[Too large to print fully. First 10 elements:]" << std::endl;
        for (size_t i = 0; i < std::min((size_t)10, total_elements); ++i) {
            std::cout << std::fixed << std::setprecision(2) << tensor[i] << " ";
        }
        std::cout << (total_elements > 10 ? "..." : "") << std::endl;
        return;
    }


    // For simplicity, printing up to 3D. Higher dimensions will be flattened.
    if (shape.size() == 1) {
        for (size_t i = 0; i < shape[0]; ++i) {
            std::cout << std::fixed << std::setprecision(2) << tensor[i] << " ";
        }
        std::cout << std::endl;
    } else if (shape.size() == 2) {
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                std::cout << std::fixed << std::setprecision(2) << tensor[i * shape[1] + j] << " ";
            }
            std::cout << std::endl;
        }
    } else if (shape.size() == 3) {
        for (size_t i = 0; i < shape[0]; ++i) {
            std::cout << "Slice " << i << ":" << std::endl;
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    std::cout << std::fixed << std::setprecision(2) << tensor[i * shape[1] * shape[2] + j * shape[2] + k] << " ";
                }
                std::cout << std::endl;
            }
            if (i < shape[0] -1) std::cout << std::endl;
        }
    } else { // Flattened print for > 3D
         for (size_t i = 0; i < total_elements; ++i) {
            std::cout << std::fixed << std::setprecision(2) << tensor[i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// CPU implementation for verification
void min_reduction_dimension_cpu(
    const float* input,
    int dim_to_reduce,
    float* output_cpu,
    const std::vector<size_t>& shape_vec,
    const std::vector<size_t>& output_shape_vec
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
    for (size_t b = 0; b < before_size; ++b) {
        for (size_t a = 0; a < after_size; ++a) {
            float min_val = FLT_MAX;
            for (size_t d = 0; d < dim_size; ++d) {
                size_t input_idx = (b * dim_size + d) * after_size + a;
                min_val = std::min(min_val, input[input_idx]);
            }
            output_cpu[output_idx_counter++] = min_val;
        }
    }
}


void run_test(const std::vector<size_t>& shape, int dim_to_reduce) {
    std::cout << "-----------------------------------------------------" << std::endl;
    std::cout << "Testing with shape: ";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << (i == shape.size() - 1 ? "" : "x");
    }
    std::cout << ", reducing dimension " << dim_to_reduce << std::endl;

    size_t total_input_elements = 1;
    for (size_t s : shape) total_input_elements *= s;

    if (total_input_elements == 0) {
        std::cout << "Input tensor is empty. Skipping test." << std::endl;
        std::cout << "-----------------------------------------------------" << std::endl << std::endl;
        return;
    }
    
    if (dim_to_reduce < 0 || dim_to_reduce >= shape.size()) {
        std::cerr << "Invalid dimension to reduce: " << dim_to_reduce << std::endl;
        std::cout << "-----------------------------------------------------" << std::endl << std::endl;
        return;
    }


    std::vector<float> h_input(total_input_elements);
    std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);
    std::generate(h_input.begin(), h_input.end(), [&]() { return dist(rng); });

    // Calculate output shape and size
    std::vector<size_t> output_shape;
    size_t total_output_elements = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i == dim_to_reduce) {
            if (shape[i] == 0) { // Cannot reduce a zero-sized dimension meaningfully in this context
                 std::cout << "Dimension " << dim_to_reduce << " has size 0. Skipping test." << std::endl;
                 std::cout << "-----------------------------------------------------" << std::endl << std::endl;
                 return;
            }
            // The reduced dimension effectively disappears or becomes 1.
            // For this test, we'll model it as disappearing.
        } else {
            output_shape.push_back(shape[i]);
            total_output_elements *= shape[i];
        }
    }
    if (output_shape.empty() && total_input_elements > 0) { // Reduction to a scalar
        output_shape.push_back(1); // Represent scalar as a 1-element array
        total_output_elements = 1;
    }
     if (total_output_elements == 0 && total_input_elements > 0 && shape[dim_to_reduce] > 0) {
        // This can happen if a non-reduced dimension is 0.
        // e.g. shape {2,0,3} reduce dim 0 -> output shape {0,3} -> total_output_elements = 0
        std::cout << "Output tensor is empty due to a zero dimension not being reduced. Skipping GPU part for safety." << std::endl;
    }


    std::vector<float> h_output_gpu(total_output_elements);
    std::vector<float> h_output_cpu(total_output_elements);

    // print_tensor(h_input.data(), shape, "Input Data");

    float *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, total_input_elements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, total_output_elements * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), total_input_elements * sizeof(float), cudaMemcpyHostToDevice));

    // GPU execution
    auto start_gpu = std::chrono::high_resolution_clock::now();
    min_reduction_dimension_cuda(d_input, dim_to_reduce, d_output, shape.data(), shape.size());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Ensure kernel completion
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_duration = end_gpu - start_gpu;

    CHECK_CUDA_ERROR(cudaMemcpy(h_output_gpu.data(), d_output, total_output_elements * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU execution for verification
    auto start_cpu = std::chrono::high_resolution_clock::now();
    min_reduction_dimension_cpu(h_input.data(), dim_to_reduce, h_output_cpu.data(), shape, output_shape);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = end_cpu - start_cpu;

    // print_tensor(h_output_gpu.data(), output_shape, "Output Data (GPU)");
    // print_tensor(h_output_cpu.data(), output_shape, "Output Data (CPU)");

    // Verification
    bool success = true;
    if (total_output_elements > 0) {
        for (size_t i = 0; i < total_output_elements; ++i) {
            if (std::abs(h_output_gpu[i] - h_output_cpu[i]) > 1e-5) {
                std::cerr << "Verification FAILED at index " << i << ": GPU = " << h_output_gpu[i]
                          << ", CPU = " << h_output_cpu[i] << std::endl;
                success = false;
                break;
            }
        }
    } else if (total_input_elements > 0 && shape[dim_to_reduce] > 0) {
        // If input had elements and reduced dim was >0, but output is empty, it implies a non-reduced dim was 0.
        // This is a valid scenario where output is empty.
        std::cout << "Output tensor is empty. Verification skipped (considered successful as expected)." << std::endl;
    }


    if (success) {
        std::cout << "Verification PASSED!" << std::endl;
    } else {
        std::cout << "Verification FAILED!" << std::endl;
        // Optionally print full tensors if failed and small enough
        if (total_input_elements <= 100) print_tensor(h_input.data(), shape, "Input Data (on fail)");
        if (total_output_elements <= 100) {
            print_tensor(h_output_gpu.data(), output_shape, "Output Data (GPU on fail)");
            print_tensor(h_output_cpu.data(), output_shape, "Output Data (CPU on fail)");
        }
    }

    std::cout << "GPU Time: " << gpu_duration.count() << " ms" << std::endl;
    std::cout << "CPU Time: " << cpu_duration.count() << " ms" << std::endl;

    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    std::cout << "-----------------------------------------------------" << std::endl << std::endl;
}


int main() {
    // Test cases
    run_test({2, 3, 4}, 0); // Reduce along the first dimension
    run_test({2, 3, 4}, 1); // Reduce along the second dimension
    run_test({2, 3, 4}, 2); // Reduce along the third dimension

    run_test({512, 512}, 0); // Larger 2D
    run_test({512, 512}, 1);

    run_test({100}, 0);      // 1D array (reduce to scalar)
    run_test({10, 1, 10}, 1); // Reduce a dimension of size 1

    run_test({10, 0, 10}, 0); // Test with a zero dimension (not reduced)
    run_test({10, 5, 10}, 1); // Reduce middle dimension
    
    run_test({2, 2, 2, 2}, 0); // 4D tensor
    run_test({2, 2, 2, 2}, 1);
    run_test({2, 2, 2, 2}, 2);
    run_test({2, 2, 2, 2}, 3);

    // Edge case: reducing a dimension that is already 1
    run_test({5, 1, 6}, 1);

    // Edge case: input shape leads to zero output elements because a *non-reduced* dimension is zero
    run_test({5, 0, 6}, 0); // output shape {0,6}, total_output_elements = 0
    run_test({5, 0, 6}, 2); // output shape {5,0}, total_output_elements = 0

    // Edge case: reducing a dimension that is zero (should be handled by checks)
    // The run_test function has a check for shape[dim_to_reduce] == 0
    run_test({5, 0, 6}, 1); // This will print "Dimension 1 has size 0. Skipping test."

    std::cout << "All tests completed." << std::endl;
    return 0;
}
