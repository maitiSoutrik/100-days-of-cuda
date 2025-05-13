#include "spectral_norm.cuh"
#include <curand.h> // For initializing random vectors
#include <cmath>    // For sqrtf

// Error checking macro for cuRAND
#define CHECK_CURAND_ERROR(call)                                                  \
    do {                                                                          \
        curandStatus_t status = call;                                             \
        if (status != CURAND_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuRAND Error at %s:%d - %d\n", __FILE__, __LINE__,   \
                    status);                                                      \
            char error_msg[256];                                                  \
            sprintf(error_msg, "cuRAND error code %d", status);                   \
            throw std::runtime_error(error_msg);                                  \
        }                                                                         \
    } while (0)

// Kernel to normalize a vector (v = v / ||v||_2)
__global__ void normalize_vector_kernel(float* vec, int n, float norm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (norm > 1e-6f) { // Avoid division by zero or very small numbers
            vec[idx] /= norm;
        }
    }
}

// Helper function to initialize a random vector on the device
void initialize_random_vector(float* d_vec, int n) {
    curandGenerator_t gen;
    CHECK_CURAND_ERROR(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    // Using a fixed seed for reproducibility in tests, can be changed
    CHECK_CURAND_ERROR(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
    CHECK_CURAND_ERROR(curandGenerateUniform(gen, d_vec, n));
    CHECK_CURAND_ERROR(curandDestroyGenerator(gen));

    // Normalize the initial random vector
    float norm_val;
    cublasHandle_t temp_handle; // Create a temporary handle for nrm2
    CHECK_CUBLAS_ERROR(cublasCreate(&temp_handle));
    CHECK_CUBLAS_ERROR(cublasSnrm2(temp_handle, n, d_vec, 1, &norm_val));
    CHECK_CUBLAS_ERROR(cublasDestroy(temp_handle));

    if (norm_val > 1e-6f) {
        int threads_per_block = 256;
        int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
        normalize_vector_kernel<<<blocks_per_grid, threads_per_block>>>(d_vec, n, norm_val);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
}


float estimate_spectral_norm(cublasHandle_t handle, const float* d_W, int m, int n,
                             float* d_u, float* d_v, int iterations) {
    // W is m x n (m rows, n columns)
    // u is m x 1
    // v is n x 1

    // Initialize v with random values and normalize
    initialize_random_vector(d_v, n);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    for (int i = 0; i < iterations; ++i) {
        // 1. u = W * v  (u_m_1 = W_m_n * v_n_1)
        // cublasSgemv(handle, trans, rows, cols, alpha, A, lda, x, incx, beta, y, incy)
        // Here W is column-major, so lda = m.
        CHECK_CUBLAS_ERROR(cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha, d_W, m, d_v, 1, &beta, d_u, 1));

        // 2. Normalize u: u = u / ||u||_2
        float norm_u;
        CHECK_CUBLAS_ERROR(cublasSnrm2(handle, m, d_u, 1, &norm_u));
        if (norm_u > 1e-6f) { // Avoid division by zero
            // Using cublasSscal for normalization: u = (1/norm_u) * u
            float inv_norm_u = 1.0f / norm_u;
            CHECK_CUBLAS_ERROR(cublasSscal(handle, m, &inv_norm_u, d_u, 1));
        }


        // 3. v = W^T * u (v_n_1 = W^T_n_m * u_m_1)
        // W is m x n, so W^T is n x m.
        // For cublasSgemv with W (m x n, col-major), W^T operation means trans = CUBLAS_OP_T.
        // When trans = CUBLAS_OP_T, rows is n, cols is m. lda is m.
        CHECK_CUBLAS_ERROR(cublasSgemv(handle, CUBLAS_OP_T, m, n, &alpha, d_W, m, d_u, 1, &beta, d_v, 1));

        // 4. Normalize v: v = v / ||v||_2
        float norm_v;
        CHECK_CUBLAS_ERROR(cublasSnrm2(handle, n, d_v, 1, &norm_v));
        if (norm_v > 1e-6f) { // Avoid division by zero
            float inv_norm_v = 1.0f / norm_v;
            CHECK_CUBLAS_ERROR(cublasSscal(handle, n, &inv_norm_v, d_v, 1));
        }
    }

    // The spectral norm sigma is ||W*v||_2 after convergence, where v is the top right singular vector.
    // Or, equivalently, u^T * W * v.
    // After the loop, u = W*v / ||W*v|| and v is normalized.
    // We need ||W*v||. We can compute W*v again and then its norm.
    CHECK_CUBLAS_ERROR(cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha, d_W, m, d_v, 1, &beta, d_u, 1));
    float spectral_norm_val;
    CHECK_CUBLAS_ERROR(cublasSnrm2(handle, m, d_u, 1, &spectral_norm_val));

    return spectral_norm_val;
}

__global__ void scale_matrix_kernel(float* matrix, int num_elements, float scalar) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        if (fabsf(scalar) > 1e-6f) { // Avoid division by zero or very small numbers
            matrix[idx] /= scalar;
        }
    }
}

void spectral_normalize_matrix(cublasHandle_t handle, float* d_W_in_out, int m, int n,
                               float* d_u, float* d_v, int iterations) {
    float sigma = estimate_spectral_norm(handle, d_W_in_out, m, n, d_u, d_v, iterations);

    if (sigma > 1e-6f) { // Avoid division by zero or very small sigma
        int num_elements = m * n;
        int threads_per_block = 256;
        int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;
        scale_matrix_kernel<<<blocks_per_grid, threads_per_block>>>(d_W_in_out, num_elements, sigma);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Ensure kernel completion
    }
    // If sigma is too small, the matrix is effectively zero, no need to scale.
}
