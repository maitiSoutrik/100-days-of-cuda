#include "muon_optimization.cuh"
#include <cmath> // For sqrtf on host, fabsf
#include <vector>
#include <numeric> // For std::iota
#include <algorithm> // For std::accumulate for host sum

// Helper function to print matrix from device (for debugging)
void print_matrix_device(const float* d_matrix, int rows, int cols, const char* label) {
    std::vector<float> h_matrix(rows * cols);
    CHECK_CUDA_ERROR(cudaMemcpy(h_matrix.data(), d_matrix, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
    print_matrix_host(h_matrix.data(), rows, cols, label);
}

// Helper function to print matrix from host
void print_matrix_host(const float* h_matrix, int rows, int cols, const char* label) {
    printf("%s (%d x %d):\n", label, rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%8.4f ", h_matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Kernel to initialize a matrix with a specific value
__global__ void initialize_matrix_kernel(float* matrix, int rows, int cols, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) {
        matrix[idy * cols + idx] = val;
    }
}

// Kernel to initialize an NxN identity matrix
__global__ void initialize_identity_matrix_kernel(float* matrix, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < N && idy < N) {
        matrix[idy * N + idx] = (idx == idy) ? 1.0f : 0.0f;
    }
}


// Kernel for matrix transposition: output[j * rows + i] = input[i * cols + j]
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Iterate over columns of input
    int idy = blockIdx.y * blockDim.y + threadIdx.y; // Iterate over rows of input

    if (idx < cols && idy < rows) {
        output[idx * rows + idy] = input[idy * cols + idx];
    }
}

// Kernel for matrix multiplication: C = A * B
// A: A_rows x A_cols
// B: A_cols x B_cols (B_rows = A_cols)
// C: A_rows x B_cols
__global__ void matrix_multiply_kernel(const float* A, const float* B, float* C, int A_rows, int A_cols, int B_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A_rows && col < B_cols) {
        float sum = 0.0f;
        for (int k = 0; k < A_cols; ++k) {
            sum += A[row * A_cols + k] * B[k * B_cols + col];
        }
        C[row * B_cols + col] = sum;
    }
}

// Kernel for element-wise scalar multiplication: matrix[i] *= scalar
__global__ void matrix_scalar_multiply_kernel(float* matrix, float scalar, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) {
        matrix[idy * cols + idx] *= scalar;
    }
}

// Kernel for element-wise matrix addition: C = A + B
__global__ void matrix_add_kernel(const float* A, const float* B, float* C, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) {
        C[idy * cols + idx] = A[idy * cols + idx] + B[idy * cols + idx];
    }
}

// Kernel for element-wise matrix subtraction: C = A - B
__global__ void matrix_elementwise_subtract_kernel(const float* A, const float* B, float* C, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) {
        C[idy * cols + idx] = A[idy * cols + idx] - B[idy * cols + idx];
    }
}

// Kernel for matrix copy: destination = source
__global__ void matrix_copy_kernel(const float* source, float* destination, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) {
        destination[idy * cols + idx] = source[idy * cols + idx];
    }
}

// Kernel to calculate sum of squares of matrix elements (part of Frobenius norm)
// Uses shared memory for reduction within a block.
// d_result must be pre-allocated to hold (gridDim.x) partial sums.
__global__ void frobenius_norm_squared_kernel(const float* matrix, float* d_block_sums, int N) {
    extern __shared__ float sdata[]; // Shared memory for partial sums within a block

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (i < N) {
        sdata[tid] = matrix[i] * matrix[i];
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        d_block_sums[blockIdx.x] = sdata[0];
    }
}

// Kernel for element-wise scalar division: matrix[i] /= scalar
__global__ void matrix_scalar_divide_kernel(float* matrix, float scalar, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) {
        if (fabsf(scalar) > 1e-9f) { // Avoid division by zero
             matrix[idy * cols + idx] /= scalar;
        } else {
             matrix[idy * cols + idx] = 0.0f; // Or handle as error
        }
    }
}


// Host function to orchestrate Newton-Schulz iteration
// O_k+1 = 1.5 * O_k - 0.5 * O_k * O_k^T * O_k  (if rows <= cols, for semi-orthogonality O*O^T = I)
// O_k+1 = 1.5 * O_k - 0.5 * O_k^T * O_k * O_k  (if rows > cols, for semi-orthogonality O^T*O = I)
// For simplicity, we'll assume rows <= cols for this example.
// The original paper mentions O_k+1 = O_k * (1.5 * I - 0.5 * O_k^T * O_k)
// which is equivalent to O_k+1 = 1.5 * O_k - 0.5 * O_k * O_k^T * O_k if O_k is already somewhat close to orthogonal.
// We will implement: O_k+1 = O_k * ( (3/2)I - (1/2)O_k^T O_k )
// Or more directly: O_k+1 = (3/2)O_k - (1/2)O_k (O_k^T O_k)
// Let's use the form: O_{k+1} = (3/2)O_k - (1/2) * O_k * (O_k^T * O_k) if rows >= cols (tall matrix, O^T O approx I_cols)
// Or O_{k+1} = (3/2)O_k - (1/2) * (O_k * O_k^T) * O_k if rows < cols (wide matrix, O O^T approx I_rows)

void newton_schulz_iteration_device(
    const float* d_G_in,    // Input matrix (rows x cols)
    float* d_G_out,         // Output matrix (rows x cols), can be same as d_G_in if modified in-place after copy
    int rows,
    int cols,
    int num_ns_iterations,
    float* d_O,             // Temporary for O_k (rows x cols)
    float* d_O_T,           // Temporary for O_k^T (cols x rows)
    float* d_prod1,         // Temporary for (O_k^T * O_k) (cols x cols) or (O_k * O_k^T) (rows x rows)
    float* d_prod2,         // Temporary for O_k * (O_k^T * O_k) or (O_k * O_k^T) * O_k (rows x cols)
    float* d_block_sums     // For Frobenius norm reduction (size = num_blocks for reduction)
) {
    dim3 threadsPerBlock(16, 16); // General purpose
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 1. Initial normalization: O_0 = G_in / ||G_in||_F
    // Calculate Frobenius norm squared of G_in
    int N = rows * cols;
    int norm_threads = 256; // Threads per block for reduction
    int norm_blocks = (N + norm_threads - 1) / norm_threads;
    
    frobenius_norm_squared_kernel<<<norm_blocks, norm_threads, norm_threads * sizeof(float)>>>(d_G_in, d_block_sums, N);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    std::vector<float> h_block_sums(norm_blocks);
    CHECK_CUDA_ERROR(cudaMemcpy(h_block_sums.data(), d_block_sums, norm_blocks * sizeof(float), cudaMemcpyDeviceToHost));
    
    float norm_sq = 0.0f;
    for(float val : h_block_sums) {
        norm_sq += val;
    }
    float norm_G_in = sqrtf(norm_sq);

    // Copy G_in to O and then normalize O
    matrix_copy_kernel<<<numBlocks, threadsPerBlock>>>(d_G_in, d_O, rows, cols);
    CHECK_CUDA_ERROR(cudaGetLastError());

    if (fabsf(norm_G_in) > 1e-9f) { // Avoid division by zero
        matrix_scalar_divide_kernel<<<numBlocks, threadsPerBlock>>>(d_O, norm_G_in, rows, cols);
        CHECK_CUDA_ERROR(cudaGetLastError());
    } else {
        // Handle zero matrix case - perhaps initialize O to something else or return G_in
        // For now, if norm is zero, O will be zero. NS won't change it.
        // Or, if G_in is zero, G_out should also be zero.
        matrix_copy_kernel<<<numBlocks, threadsPerBlock>>>(d_G_in, d_G_out, rows, cols);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        // printf("Input matrix G_in has zero Frobenius norm. Skipping NS iterations.\n");
        return;
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    // printf("Initial O_0 (normalized G_in):\n");
    // print_matrix_device(d_O, rows, cols, "O_0");


    // Newton-Schulz Iterations:
    // O_{k+1} = 1.5 * O_k - 0.5 * term
    // where term = O_k * (O_k^T * O_k)  if rows >= cols (tall or square, O^T O is smaller)
    // or    term = (O_k * O_k^T) * O_k  if rows < cols  (wide, O O^T is smaller)

    bool tall_or_square = (rows >= cols);

    for (int iter = 0; iter < num_ns_iterations; ++iter) {
        // printf("NS Iteration %d\n", iter + 1);

        // Calculate O_k^T
        dim3 transposeBlocks((rows + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                             (cols + threadsPerBlock.y - 1) / threadsPerBlock.y);
        matrix_transpose_kernel<<<transposeBlocks, threadsPerBlock>>>(d_O, d_O_T, rows, cols);
        CHECK_CUDA_ERROR(cudaGetLastError());
        // print_matrix_device(d_O_T, cols, rows, "O_T");

        if (tall_or_square) { // O is rows x cols, O_T is cols x rows. O_T * O is cols x cols
            // d_prod1 = O_k^T * O_k  (cols x cols)
            dim3 prod1Blocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                             (cols + threadsPerBlock.y - 1) / threadsPerBlock.y);
            matrix_multiply_kernel<<<prod1Blocks, threadsPerBlock>>>(d_O_T, d_O, d_prod1, cols, rows, cols);
            CHECK_CUDA_ERROR(cudaGetLastError());
            // print_matrix_device(d_prod1, cols, cols, "O_T * O");

            // d_prod2 = O_k * (O_k^T * O_k) (rows x cols)
            dim3 prod2Blocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                             (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
            matrix_multiply_kernel<<<prod2Blocks, threadsPerBlock>>>(d_O, d_prod1, d_prod2, rows, cols, cols);
            CHECK_CUDA_ERROR(cudaGetLastError());
            // print_matrix_device(d_prod2, rows, cols, "O * (O_T * O)");
        } else { // Wide matrix: rows < cols. O * O_T is rows x rows
            // d_prod1 = O_k * O_k^T (rows x rows)
            dim3 prod1Blocks((rows + threadsPerBlock.x - 1) / threadsPerBlock.x,
                             (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
            matrix_multiply_kernel<<<prod1Blocks, threadsPerBlock>>>(d_O, d_O_T, d_prod1, rows, cols, rows);
            CHECK_CUDA_ERROR(cudaGetLastError());
            // print_matrix_device(d_prod1, rows, rows, "O * O_T");
            
            // d_prod2 = (O_k * O_k^T) * O_k (rows x cols)
            dim3 prod2Blocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                             (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
            matrix_multiply_kernel<<<prod2Blocks, threadsPerBlock>>>(d_prod1, d_O, d_prod2, rows, rows, cols);
            CHECK_CUDA_ERROR(cudaGetLastError());
            // print_matrix_device(d_prod2, rows, cols, "(O * O_T) * O");
        }

        // O_k+1 = 1.5 * O_k - 0.5 * d_prod2
        // Step 1: Scale O_k by 1.5 (store in d_G_out temporarily or a new temp if d_O needs to be preserved for d_prod2)
        // For in-place on d_O:
        //    Copy d_O to d_G_out (this will be the 1.5 * O_k part)
        //    Scale d_G_out by 1.5
        //    Scale d_prod2 by -0.5
        //    Add d_prod2 to d_G_out. d_G_out is now O_k+1
        //    Copy d_G_out back to d_O for next iteration

        matrix_copy_kernel<<<numBlocks, threadsPerBlock>>>(d_O, d_G_out, rows, cols); // d_G_out = O_k
        CHECK_CUDA_ERROR(cudaGetLastError());
        matrix_scalar_multiply_kernel<<<numBlocks, threadsPerBlock>>>(d_G_out, 1.5f, rows, cols); // d_G_out = 1.5 * O_k
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        // d_prod2 is already O_k * O_k^T * O_k or (O_k * O_k^T) * O_k
        // We need -0.5 * d_prod2. We can scale d_prod2 by -0.5 then add.
        matrix_scalar_multiply_kernel<<<numBlocks, threadsPerBlock>>>(d_prod2, -0.5f, rows, cols); // d_prod2 = -0.5 * term
        CHECK_CUDA_ERROR(cudaGetLastError());

        matrix_add_kernel<<<numBlocks, threadsPerBlock>>>(d_G_out, d_prod2, d_O, rows, cols); // d_O (new O_k+1) = 1.5 * O_k - 0.5 * term
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        // printf("After NS iteration %d:\n", iter + 1);
        // print_matrix_device(d_O, rows, cols, "O_k+1");
    }

    // Copy the final O_k (which is in d_O) to d_G_out
    matrix_copy_kernel<<<numBlocks, threadsPerBlock>>>(d_O, d_G_out, rows, cols);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
