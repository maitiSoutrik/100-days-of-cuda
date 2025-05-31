#include "negative_cosine_similarity.cuh"
#include <iostream>
#include <vector>
#include <numeric>   // For std::iota (not used here, but good include for general data init)
#include <algorithm> // For std::generate, std::transform (not used here)
#include <iomanip>   // For std::fixed, std::setprecision
#include <cmath>     // For fabs, sqrtf, fmaxf in CPU version

// Helper function to print vectors row by row
void print_vector_data(const std::string& name, const std::vector<float>& vec, size_t n, size_t d) {
    std::cout << name << " (Rows: " << n << ", Dim: " << d << "):" << std::endl;
    for (size_t i = 0; i < n; ++i) {
        std::cout << "  Vec " << i << ": [";
        for (size_t j = 0; j < d; ++j) {
            std::cout << std::fixed << std::setprecision(4) << vec[i * d + j] << (j == d - 1 ? "" : ", ");
        }
        std::cout << "]" << std::endl;
    }
}

// Helper function to print output vector
void print_output_data(const std::string& name, const std::vector<float>& vec, size_t n) {
    std::cout << name << ":" << std::endl;
    for (size_t i = 0; i < n; ++i) {
        std::cout << "  Output for pair " << i << ": " << std::fixed << std::setprecision(6) << vec[i] << std::endl;
    }
}

// CPU implementation for verification
void cosine_similarity_cpu(const std::vector<float>& predictions, const std::vector<float>& targets, std::vector<float>& output, size_t n, size_t d) {
    const float eps = 1e-8f;
    for (size_t i = 0; i < n; ++i) {
        float dot = 0.0f;
        float norm_pred = 0.0f;
        float norm_target = 0.0f;
        size_t offset = i * d;

        for (size_t j = 0; j < d; ++j) {
            float p = predictions[offset + j];
            float t = targets[offset + j];
            dot += p * t;
            norm_pred += p * p;
            norm_target += t * t;
        }
        norm_pred = sqrtf(norm_pred);
        norm_target = sqrtf(norm_target);
        
        float effective_norm_pred = fmaxf(eps, norm_pred);
        float effective_norm_target = fmaxf(eps, norm_target);
        float denom = effective_norm_pred * effective_norm_target;
        
        float cosine_sim = 0.0f;
        if (denom > eps * eps / 2.0f) { // Consistent with kernel's robustness
            cosine_sim = dot / denom;
        } else if (norm_pred == 0.0f && norm_target == 0.0f) {
            cosine_sim = 0.0f; // Or 1.0f if defining sim(0,0) as 1. Sticking to 0.
        }
        // Clamp cosine_sim to [-1, 1] due to potential floating point inaccuracies
        cosine_sim = fmaxf(-1.0f, fminf(1.0f, cosine_sim));

        output[i] = 1.0f - cosine_sim;
    }
}


int main() {
    size_t n = 5; // Number of vector pairs
    size_t d = 3; // Dimension of each vector

    std::vector<float> h_predictions(n * d);
    std::vector<float> h_targets(n * d);
    std::vector<float> h_output_gpu(n);
    std::vector<float> h_output_cpu(n);

    // Initialize with some example data
    // Pair 0: Identical vectors (Expected 1 - cos_sim = 1 - 1 = 0)
    h_predictions[0*d+0] = 1.0f; h_predictions[0*d+1] = 2.0f; h_predictions[0*d+2] = 3.0f;
    h_targets[0*d+0]   = 1.0f; h_targets[0*d+1]   = 2.0f; h_targets[0*d+2]   = 3.0f;

    // Pair 1: Orthogonal vectors (Expected 1 - cos_sim = 1 - 0 = 1)
    h_predictions[1*d+0] = 1.0f; h_predictions[1*d+1] = 0.0f; h_predictions[1*d+2] = 0.0f;
    h_targets[1*d+0]   = 0.0f; h_targets[1*d+1]   = 1.0f; h_targets[1*d+2]   = 0.0f;

    // Pair 2: Opposite vectors (Expected 1 - cos_sim = 1 - (-1) = 2)
    h_predictions[2*d+0] = 1.0f; h_predictions[2*d+1] = 1.0f; h_predictions[2*d+2] = 1.0f;
    h_targets[2*d+0]   = -1.0f;h_targets[2*d+1]   = -1.0f;h_targets[2*d+2]   = -1.0f;

    // Pair 3: General vectors
    h_predictions[3*d+0] = 0.5f; h_predictions[3*d+1] = -0.5f;h_predictions[3*d+2] = 1.0f;
    h_targets[3*d+0]   = 0.2f; h_targets[3*d+1]   = 0.8f; h_targets[3*d+2]   = -0.3f;
    // Dot: (0.5*0.2) + (-0.5*0.8) + (1.0*-0.3) = 0.1 - 0.4 - 0.3 = -0.6
    // Norm P: sqrt(0.25 + 0.25 + 1) = sqrt(1.5) = 1.2247
    // Norm T: sqrt(0.04 + 0.64 + 0.09) = sqrt(0.77) = 0.8775
    // Cos_sim = -0.6 / (1.2247 * 0.8775) = -0.6 / 1.0747 = -0.5583
    // Output = 1 - (-0.5583) = 1.5583

    // Pair 4: One vector is zero (Expected 1 - cos_sim = 1 - 0 = 1)
    h_predictions[4*d+0] = 0.0f; h_predictions[4*d+1] = 0.0f; h_predictions[4*d+2] = 0.0f;
    h_targets[4*d+0]   = 1.0f; h_targets[4*d+1]   = 2.0f; h_targets[4*d+2]   = 3.0f;


    print_vector_data("Predictions (Host)", h_predictions, n, d);
    print_vector_data("Targets (Host)", h_targets, n, d);

    float *d_predictions, *d_targets, *d_output;
    cudaError_t err;

    err = cudaMalloc((void**)&d_predictions, n * d * sizeof(float)); CHECK_CUDA_ERROR(err);
    err = cudaMalloc((void**)&d_targets, n * d * sizeof(float));     CHECK_CUDA_ERROR(err);
    err = cudaMalloc((void**)&d_output, n * sizeof(float));          CHECK_CUDA_ERROR(err);

    err = cudaMemcpy(d_predictions, h_predictions.data(), n * d * sizeof(float), cudaMemcpyHostToDevice); CHECK_CUDA_ERROR(err);
    err = cudaMemcpy(d_targets, h_targets.data(), n * d * sizeof(float), cudaMemcpyHostToDevice);         CHECK_CUDA_ERROR(err);

    launch_cosine_similarity_kernel(d_predictions, d_targets, d_output, n, d);
    // Synchronize device to ensure kernel completion before checking errors or copying data back
    err = cudaDeviceSynchronize(); CHECK_CUDA_ERROR(err); 

    // Check for errors after kernel execution (optional if launch_cosine_similarity_kernel checks, but good practice here)
    cudaError_t kernel_err = cudaGetLastError(); // This will pick up any async errors from the kernel
    if (kernel_err != cudaSuccess) {
        fprintf(stderr, "CUDA error after kernel execution in main: %s\n", cudaGetErrorString(kernel_err));
    }

    err = cudaMemcpy(h_output_gpu.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost); CHECK_CUDA_ERROR(err);

    print_output_data("GPU Output (1.0 - Cosine Similarity)", h_output_gpu, n);

    // CPU calculation for verification
    cosine_similarity_cpu(h_predictions, h_targets, h_output_cpu, n, d);
    print_output_data("CPU Output (1.0 - Cosine Similarity) for Verification", h_output_cpu, n);

    // Verification
    bool success = true;
    float tolerance = 1e-5f; // Adjusted tolerance
    for (size_t i = 0; i < n; ++i) {
        if (fabs(h_output_gpu[i] - h_output_cpu[i]) > tolerance) {
            std::cerr << "Mismatch at index " << i << ": GPU = " << h_output_gpu[i]
                      << ", CPU = " << h_output_cpu[i] 
                      << ", Diff = " << fabs(h_output_gpu[i] - h_output_cpu[i]) << std::endl;
            success = false;
        }
    }

    if (success) {
        std::cout << "Verification Successful: GPU and CPU results match within tolerance." << std::endl;
    } else {
        std::cout << "Verification Failed: GPU and CPU results differ." << std::endl;
    }

    cudaFree(d_predictions);
    cudaFree(d_targets);
    cudaFree(d_output);

    return success ? 0 : 1;
}
