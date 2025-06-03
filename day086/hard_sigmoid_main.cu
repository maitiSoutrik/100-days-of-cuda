// day086/hard_sigmoid_main.cu
#include "hard_sigmoid.cuh"
#include "common_utils.h" // For error checking macros
#include <vector>
#include <iostream>
#include <iomanip> // For std::fixed and std::setprecision
#include <cmath>   // For std::abs

void print_matrix(const std::vector<float>& matrix, size_t n, size_t m, const std::string& title) {
    std::cout << title << ":" << std::endl;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            std::cout << std::fixed << std::setprecision(3) << matrix[i * m + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    size_t n = 4;
    size_t m = 5;
    size_t total_elements = n * m;

    std::vector<float> h_input(total_elements);
    std::vector<float> h_output(total_elements);

    // Initialize input data: -10, -9, ..., 0, ..., 9
    for (size_t i = 0; i < total_elements; ++i) {
        h_input[i] = static_cast<float>(i) - 10.0f; 
    }

    print_matrix(h_input, n, m, "Input Matrix");

    // Call the CUDA solution
    hard_sigmoid_solution(h_input.data(), h_output.data(), n, m);
    
    // Check for any CUDA errors that might have occurred within hard_sigmoid_solution 
    // if not handled by exit (though common_utils.h currently uses exit)
    CHECK_LAST_CUDA_ERROR(); 

    print_matrix(h_output, n, m, "Output Matrix (Hard Sigmoid)");

    // Verification (simple check)
    bool success = true;
    for (size_t i = 0; i < total_elements; ++i) {
        float x = h_input[i];
        float expected_output;
        if (x <= -3.0f) {
            expected_output = 0.0f;
        } else if (x >= 3.0f) {
            expected_output = 1.0f;
        } else {
            expected_output = (x + 3.0f) / 6.0f;
        }
        if (std::abs(h_output[i] - expected_output) > 1e-5) {
            std::cerr << "Verification failed at index " << i << ": input=" << x
                      << ", output=" << h_output[i] << ", expected=" << expected_output << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Verification successful!" << std::endl;
    } else {
        std::cout << "Verification failed." << std::endl;
    }

    return success ? 0 : 1;
}
