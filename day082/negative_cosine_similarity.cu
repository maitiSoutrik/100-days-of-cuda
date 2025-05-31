#include "negative_cosine_similarity.cuh"
#include <stdio.h> // For printf in kernel error reporting
#include <math.h>  // For sqrtf, fmaxf

__global__ void cosine_similarity_kernel(const float* predictions, const float* targets, float* output, size_t n, size_t d) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float dot = 0.0f;
        float norm_pred = 0.0f;
        float norm_target = 0.0f;
        size_t offset = idx * d;

        for (size_t j = 0; j < d; j++) {
            float p = predictions[offset + j];
            float t = targets[offset + j];
            dot += p * t;
            norm_pred += p * p;
            norm_target += t * t;
        }

        norm_pred = sqrtf(norm_pred);
        norm_target = sqrtf(norm_target);

        const float eps = 1e-8f; // Epsilon to prevent division by zero
        
        // Ensure denominators are at least eps to avoid issues with zero vectors
        float effective_norm_pred = fmaxf(eps, norm_pred);
        float effective_norm_target = fmaxf(eps, norm_target);
        float denom = effective_norm_pred * effective_norm_target;
        
        float cosine_sim = 0.0f; 
        
        // Only divide if denominator is meaningfully non-zero.
        // If both norms were originally zero, denom will be eps*eps.
        // If one norm was zero, denom will be eps * other_norm.
        // If both norms non-zero, denom = norm_pred * norm_target.
        // This check ensures we don't divide by a very small number if not intended.
        if (denom > eps * eps / 2.0f) { // A bit more robust than just denom > 0 for floating point
             cosine_sim = dot / denom;
        } else if (norm_pred == 0.0f && norm_target == 0.0f) {
            // Special case: similarity of two zero vectors. Can be 1 (identical) or 0 (undefined).
            // Let's define it as 1 (perfectly similar) for 1-cos_sim to be 0.
            // Or, if we want 1-cos_sim to be 1 (neutral), set cos_sim = 0.
            // The original code implies cos_sim = 0 if denom is too small.
            cosine_sim = 0.0f; // Consistent with original logic if denom is too small
        }


        // Output is 1.0 - cosine_similarity, often called cosine distance
        output[idx] = 1.0f - cosine_sim;
    }
}

extern "C" void launch_cosine_similarity_kernel(const float* predictions, const float* targets, float* output, size_t n, size_t d) {
    if (n == 0) { // Handle empty input gracefully
        return;
    }
    size_t total_vectors = n;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_vectors + threadsPerBlock - 1) / threadsPerBlock;

    cosine_similarity_kernel<<<blocksPerGrid, threadsPerBlock>>>(predictions, targets, output, n, d);

    // The user's original code had a printf here for cudaGetLastError().
    // This can be useful for debugging during development.
    // For a library, error checking is typically propagated or handled by the caller.
    // We'll keep it as per the .clinerules to follow existing patterns if they include such checks.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // This printf will be visible if the main application is run from a console.
        printf("CUDA Error after cosine_similarity_kernel launch: %s\n", cudaGetErrorString(err));
        // Depending on project policy, might re-throw, return error code, or just log.
    }
}
