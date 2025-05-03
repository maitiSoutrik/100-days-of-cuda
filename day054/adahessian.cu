#include "adahessian.h"
#include <math.h> // For sqrtf

// AdaHessian update kernel Implementation
__global__ void adaHessianUpdateKernel(
    float* theta,
    const float* grad,
    const float* gradPerturbed,
    float* m,
    float* v,
    const float lr,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float delta,
    int N
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        // Approximate the Hessian diagonal using finite differences:
        // Ensure delta is not zero to avoid division by zero
        float h_diag = (delta != 0.0f) ? (gradPerturbed[idx] - grad[idx]) / delta : 0.0f;

        // Update first moment (gradient) estimate:
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad[idx];

        // Update second moment (squared Hessian diag) estimate:
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * (h_diag * h_diag);

        // Calculate denominator for update
        float denom = sqrtf(v[idx]) + epsilon;

        // Update parameters using the AdaHessian rule:
        // Check denom is not zero before division
        if (denom != 0.0f) {
             theta[idx] -= lr * m[idx] / denom;
        }
    }
}

// Add a dummy function to ensure this compiles as a separate translation unit if needed
// though CMake should handle .cu files correctly.
void adahessian_dummy_link_function() {}
