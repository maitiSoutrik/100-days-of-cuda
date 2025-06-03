#include "tensor_matrix_mult.cuh"
#include <iostream>
#include <vector>
#include <numeric>   // For std::iota
#include <algorithm> // For std::fill, std::equal
#include <iomanip>   // For std::fixed, std::setprecision
#include <chrono>    // For timing

// Helper function to initialize a vector with sequential values
template<typename T>
void initialize_data(std::vector<T>& data, T start_val = 0) {
    std::iota(data.begin(), data.end(), start_val);
}

// Helper function to print a slice of a tensor/matrix for verification
template<typename T>
void print_slice(const std::vector<T>& data, size_t max_elements_to_print = 16) {
    size_t count = 0;
    for (const auto& val : data) {
        if (count >= max_elements_to_print) {
            std::cout << "...";
            break;
        }
        std::cout << val << " ";
        count++;
    }
    std::cout << std::endl;
}

// CPU implementation for verification
void tensor_matrix_multiply_cpu(
    const std::vector<float>& A_host,
    const std::vector<float>& B_host,
    std::vector<float>& C_host,
    size_t B_dim, size_t I_dim, size_t J_dim, size_t L_dim, size_t K_dim) {

    C_host.assign(B_dim * I_dim * J_dim * K_dim, 0.0f);

    for (size_t b = 0; b < B_dim; ++b) {
        for (size_t i = 0; i < I_dim; ++i) {
            for (size_t j = 0; j < J_dim; ++j) {
                for (size_t k = 0; k < K_dim; ++k) {
                    float sum = 0.0f;
                    for (size_t l = 0; l < L_dim; ++l) {
                        // A_idx = b*I*J*L + i*J*L + j*L + l
                        size_t a_idx = ((b * I_dim + i) * J_dim + j) * L_dim + l;
                        // B_idx = l*K + k
                        size_t b_idx = l * K_dim + k;
                        sum += A_host[a_idx] * B_host[b_idx];
                    }
                    // C_idx = b*I*J*K + i*J*K + j*K + k
                    size_t c_idx = (((b * I_dim + i) * J_dim + j) * K_dim + k);
                    C_host[c_idx] = sum;
                }
            }
        }
    }
}


int main() {
    // Define tensor dimensions
    // Example: A is (2, 3, 4, 5), B is (5, 6) -> C is (2, 3, 4, 6)
    size_t B_dim = 2;  // Batch
    size_t I_dim = 3;  // Input Channels / Height
    size_t J_dim = 4;  // Width
    size_t L_dim = 5;  // Contraction dimension (common between A's last and B's first)
    size_t K_dim = 6;  // Output Channels / B's second dimension

    std::cout << "Tensor-Matrix Multiplication" << std::endl;
    std::cout << "Dimensions:" << std::endl;
    std::cout << "  A: (" << B_dim << ", " << I_dim << ", " << J_dim << ", " << L_dim << ")" << std::endl;
    std::cout << "  B: (" << L_dim << ", " << K_dim << ")" << std::endl;
    std::cout << "  C: (" << B_dim << ", " << I_dim << ", " << J_dim << ", " << K_dim << ")" << std::endl;
    std::cout << "------------------------------------" << std::endl;

    // Calculate sizes
    size_t A_size = B_dim * I_dim * J_dim * L_dim;
    size_t B_size = L_dim * K_dim;
    size_t C_size = B_dim * I_dim * J_dim * K_dim;

    // Host data
    std::vector<float> h_A(A_size);
    std::vector<float> h_B(B_size);
    std::vector<float> h_C_gpu(C_size); // Result from GPU
    std::vector<float> h_C_cpu(C_size); // Result from CPU for verification

    // Initialize host data (e.g., with sequential numbers for simplicity)
    initialize_data(h_A, 1.0f);
    initialize_data(h_B, 0.5f);

    // Device data pointers
    float *d_A, *d_B, *d_C;

    // Allocate memory on GPU
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A, A_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B, B_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_C, C_size * sizeof(float)));

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A.data(), A_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B.data(), B_size * sizeof(float), cudaMemcpyHostToDevice));

    // GPU computation
    std::cout << "Performing GPU computation..." << std::endl;
    auto start_gpu = std::chrono::high_resolution_clock::now();

    tensor_matrix_multiply(d_A, d_B, d_C, B_dim, I_dim, J_dim, L_dim, K_dim);
    CHECK_KERNEL_LAUNCH(); // Check for kernel launch errors and synchronize

    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_duration = end_gpu - start_gpu;
    std::cout << "GPU computation time: " << std::fixed << std::setprecision(3)
              << gpu_duration.count() << " ms" << std::endl;

    // Copy result from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_C_gpu.data(), d_C, C_size * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU computation for verification
    std::cout << "Performing CPU computation for verification..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    tensor_matrix_multiply_cpu(h_A, h_B, h_C_cpu, B_dim, I_dim, J_dim, L_dim, K_dim);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = end_cpu - start_cpu;
    std::cout << "CPU computation time: " << std::fixed << std::setprecision(3)
              << cpu_duration.count() << " ms" << std::endl;

    // Verification
    bool mismatch = false;
    float tolerance = 1e-4f; // Tolerance for floating point comparison
    for (size_t i = 0; i < C_size; ++i) {
        if (std::abs(h_C_gpu[i] - h_C_cpu[i]) > tolerance) {
            std::cerr << "Mismatch at index " << i << ": GPU = " << h_C_gpu[i]
                      << ", CPU = " << h_C_cpu[i] << std::endl;
            mismatch = true;
            break; 
        }
    }

    if (!mismatch) {
        std::cout << "Verification successful: GPU and CPU results match within tolerance." << std::endl;
    } else {
        std::cout << "Verification FAILED: GPU and CPU results differ." << std::endl;
    }

    std::cout << "Sample of GPU result (first ~16 elements of C):" << std::endl;
    print_slice(h_C_gpu);
    std::cout << "Sample of CPU result (first ~16 elements of C):" << std::endl;
    print_slice(h_C_cpu);


    // Free GPU memory
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));

    std::cout << "------------------------------------" << std::endl;
    std::cout << "Day 085 execution finished." << std::endl;

    return 0;
}
