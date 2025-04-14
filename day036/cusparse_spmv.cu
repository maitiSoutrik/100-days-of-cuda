#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cusparse.h> // Include cuSPARSE header
#include <math.h>
#include <time.h>

// CUDA Error checking macro
#define CHECK_CUDA_ERROR(err) {\
    if (err != cudaSuccess) {\
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));\
        exit(EXIT_FAILURE);\
    }\
}

// cuSPARSE Error checking macro
#define CHECK_CUSPARSE_ERROR(status) {\
    if (status != CUSPARSE_STATUS_SUCCESS) {\
        fprintf(stderr, "cuSPARSE error in %s:%d: %s\n", __FILE__, __LINE__, cusparseGetErrorString(status));\
        exit(EXIT_FAILURE);\
    }\
}

// CSR format sparse matrix structure (same as day010)
typedef struct {
    int num_rows;
    int num_cols;
    int num_nonzeros;
    int *row_offsets;
    int *col_indices;
    float *values;
} CSRMatrix;

// Generate a random sparse matrix (modified from day010 to handle larger sizes better)
CSRMatrix* generateRandomSparseMatrix(int num_rows, int num_cols, float sparsity) {
    CSRMatrix *matrix = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    if (!matrix) { fprintf(stderr, "Failed to allocate memory for CSRMatrix struct\n"); exit(EXIT_FAILURE); }
    matrix->num_rows = num_rows;
    matrix->num_cols = num_cols;

    // Use long long for potentially large total elements
    long long max_possible_nonzeros = (long long)num_rows * num_cols;
    long long estimated_nonzeros_ll = (long long)(max_possible_nonzeros * sparsity);

    // Ensure estimated_nonzeros doesn't exceed INT_MAX if using int for nnz count later
    if (estimated_nonzeros_ll > INT_MAX) {
        fprintf(stderr, "Warning: Estimated non-zeros exceed INT_MAX. Clamping.\n");
        estimated_nonzeros_ll = INT_MAX;
    }
    int estimated_nonzeros = (int)estimated_nonzeros_ll;
    // Add a buffer factor (e.g., 10%) in case sparsity estimate is low or guarantees are needed
    int allocation_size = (int)(estimated_nonzeros * 1.1);
    if (allocation_size <= 0) allocation_size = num_rows; // Ensure at least one element per row possible


    matrix->row_offsets = (int*)malloc((num_rows + 1) * sizeof(int));
    matrix->col_indices = (int*)malloc(allocation_size * sizeof(int));
    matrix->values = (float*)malloc(allocation_size * sizeof(float));

    if (!matrix->row_offsets || !matrix->col_indices || !matrix->values) {
        fprintf(stderr, "Failed to allocate memory for CSR matrix arrays\n");
        free(matrix->row_offsets); free(matrix->col_indices); free(matrix->values); free(matrix);
        exit(EXIT_FAILURE);
    }

    matrix->row_offsets[0] = 0;
    int nnz = 0;
    srand(time(NULL));

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            if (((float)rand() / RAND_MAX) < sparsity) {
                if (nnz < allocation_size) {
                    matrix->col_indices[nnz] = j;
                    matrix->values[nnz] = (float)rand() / RAND_MAX * 10.0f;
                    nnz++;
                } else {
                    // This should ideally not happen with the buffer, but handle it just in case
                    fprintf(stderr, "Warning: Exceeded allocated non-zero storage. Reallocation needed (not implemented).\n");
                    // Here you might realloc, but for simplicity, we'll just stop adding elements.
                    goto end_generation; // Exit nested loops
                }
            }
        }
        matrix->row_offsets[i + 1] = nnz;
    }

end_generation:
    // If loops finished early due to allocation limit, fill remaining row_offsets
    for (int i = matrix->row_offsets[0] == 0 ? 1 : 0; i <= num_rows; ++i) {
        if(matrix->row_offsets[i] == 0 && i > 0 && matrix->row_offsets[i-1] != 0) { // If current offset is uninitialized but previous is not
           matrix->row_offsets[i] = matrix->row_offsets[i-1]; // Set it to the previous offset (or handle as needed)
        } else if (matrix->row_offsets[i] == 0 && i == 0) {
            matrix->row_offsets[i] = 0; // Ensure first element is 0
        }
         // Assuming remaining offsets should hold the final nnz count if generation stopped early
        if(matrix->row_offsets[i] == 0 && i > 0) matrix->row_offsets[i] = nnz;
    }


    matrix->num_nonzeros = nnz;

    // Optional: Ensure each row has at least one non-zero element (can skew sparsity slightly)
    // This part might need more robust memory handling if allocation_size is tight
    /*
    for (int i = 0; i < num_rows; i++) {
        if (matrix->row_offsets[i] == matrix->row_offsets[i + 1]) {
           // Complex logic to insert an element, requires careful shifting and potential realloc
           // Skipping for this example to avoid complexity with large matrices
        }
    }
    */

    // Trim allocated memory to actual size (optional, saves memory)
    int* temp_col = (int*)realloc(matrix->col_indices, nnz * sizeof(int));
    float* temp_val = (float*)realloc(matrix->values, nnz * sizeof(float));
    if (temp_col) matrix->col_indices = temp_col;
    if (temp_val) matrix->values = temp_val;


    return matrix;
}

// Free memory allocated for CSR matrix
void freeCSRMatrix(CSRMatrix *matrix) {
    if (matrix) {
        free(matrix->row_offsets);
        free(matrix->col_indices);
        free(matrix->values);
        free(matrix);
    }
}

// Print matrix statistics
void printMatrixInfo(const CSRMatrix *matrix) {
    printf("Sparse Matrix Info:\n");
    printf("  Dimensions: %d x %d\n", matrix->num_rows, matrix->num_cols);
    printf("  Non-zeros: %d\n", matrix->num_nonzeros);
    double actual_sparsity = (double)matrix->num_nonzeros / ((double)matrix->num_rows * matrix->num_cols);
    printf("  Actual Sparsity: %.6f%%\n", 100.0 * actual_sparsity);
}

// CPU implementation of SpMV for verification (same as day010)
void spmvCPU(const CSRMatrix *matrix, const float *x, float *y) {
    for (int i = 0; i < matrix->num_rows; i++) {
        float dot = 0.0f;
        int row_start = matrix->row_offsets[i];
        int row_end = matrix->row_offsets[i + 1];
        for (int j = row_start; j < row_end; j++) {
            dot += matrix->values[j] * x[matrix->col_indices[j]];
        }
        y[i] = dot;
    }
}

// Verify results by comparing CPU and GPU outputs
bool verifyResults(const float *cpu_result, const float *gpu_result, int size, float tolerance) {
    int errors = 0;
    for (int i = 0; i < size; i++) {
        float diff = fabs(cpu_result[i] - gpu_result[i]);
        // Use relative error, but handle cases where cpu_result[i] is close to zero
        float rel_err = (fabs(cpu_result[i]) > 1e-6) ? diff / fabs(cpu_result[i]) : diff;

        if (rel_err > tolerance) {
             if (errors < 10) { // Print first few errors
                fprintf(stderr, "Verification failed at index %d: CPU = %f, GPU = %f, Diff = %f, RelErr = %f\n",
                       i, cpu_result[i], gpu_result[i], diff, rel_err);
             }
             errors++;
        }
    }
     if (errors > 0) {
        fprintf(stderr, "Total verification errors: %d\n", errors);
        return false;
    }
    return true;
}


int main(int argc, char **argv) {
    // Increase matrix size for better performance comparison
    int num_rows = 20000;
    int num_cols = 20000;
    float sparsity = 0.01f; // Keep 1% sparsity

    if (argc > 1) num_rows = atoi(argv[1]);
    if (argc > 2) num_cols = atoi(argv[2]);
    if (argc > 3) sparsity = atof(argv[3]);

    printf("cuSPARSE Sparse Matrix-Vector Multiplication (SpMV)\n");
    printf("Matrix size: %d x %d, Target Sparsity: %.2f%%\n\n", num_rows, num_cols, sparsity * 100);

    // Generate sparse matrix on host
    CSRMatrix *h_matrix = generateRandomSparseMatrix(num_rows, num_cols, sparsity);
    if (!h_matrix) return EXIT_FAILURE; // Check if generation failed
    printMatrixInfo(h_matrix);
    int nnz = h_matrix->num_nonzeros; // Actual number of non-zeros

    // Generate input vector x on host
    float *h_x = (float*)malloc(num_cols * sizeof(float));
    if (!h_x) { fprintf(stderr, "Failed to allocate host vector x\n"); freeCSRMatrix(h_matrix); return EXIT_FAILURE; }
    for (int i = 0; i < num_cols; i++) {
        h_x[i] = (float)rand() / RAND_MAX;
    }

    // Allocate host output vectors
    float *h_y_cpu = (float*)malloc(num_rows * sizeof(float));
    float *h_y_gpu = (float*)malloc(num_rows * sizeof(float));
     if (!h_y_cpu || !h_y_gpu) {
        fprintf(stderr, "Failed to allocate host output vectors\n");
        free(h_x); free(h_y_cpu); free(h_y_gpu); freeCSRMatrix(h_matrix); return EXIT_FAILURE;
    }


    // --- CPU Computation (for verification and timing) ---
    printf("\nComputing SpMV on CPU...\n");
    clock_t cpu_start = clock();
    spmvCPU(h_matrix, h_x, h_y_cpu);
    clock_t cpu_end = clock();
    double cpu_time_ms = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU Execution Time: %.4f ms\n", cpu_time_ms);

    // --- GPU Computation using cuSPARSE ---
    printf("\nComputing SpMV on GPU using cuSPARSE...\n");

    cusparseHandle_t handle = NULL;
    cusparseMatDescr_t descr = NULL;
    int *d_row_offsets = NULL, *d_col_indices = NULL;
    float *d_values = NULL, *d_x = NULL, *d_y = NULL;
    cudaError_t cuda_status;
    cusparseStatus_t status;

    // Initialize cuSPARSE
    status = cusparseCreate(&handle);
    CHECK_CUSPARSE_ERROR(status);

    // Create matrix descriptor
    status = cusparseCreateMatDescr(&descr);
    CHECK_CUSPARSE_ERROR(status);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO); // 0-based indexing

    // Allocate device memory
    cuda_status = cudaMalloc((void**)&d_row_offsets, (num_rows + 1) * sizeof(int)); CHECK_CUDA_ERROR(cuda_status);
    cuda_status = cudaMalloc((void**)&d_col_indices, nnz * sizeof(int)); CHECK_CUDA_ERROR(cuda_status);
    cuda_status = cudaMalloc((void**)&d_values, nnz * sizeof(float)); CHECK_CUDA_ERROR(cuda_status);
    cuda_status = cudaMalloc((void**)&d_x, num_cols * sizeof(float)); CHECK_CUDA_ERROR(cuda_status);
    cuda_status = cudaMalloc((void**)&d_y, num_rows * sizeof(float)); CHECK_CUDA_ERROR(cuda_status);

    // Copy data from host to device
    cuda_status = cudaMemcpy(d_row_offsets, h_matrix->row_offsets, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice); CHECK_CUDA_ERROR(cuda_status);
    cuda_status = cudaMemcpy(d_col_indices, h_matrix->col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice); CHECK_CUDA_ERROR(cuda_status);
    cuda_status = cudaMemcpy(d_values, h_matrix->values, nnz * sizeof(float), cudaMemcpyHostToDevice); CHECK_CUDA_ERROR(cuda_status);
    cuda_status = cudaMemcpy(d_x, h_x, num_cols * sizeof(float), cudaMemcpyHostToDevice); CHECK_CUDA_ERROR(cuda_status);
    cuda_status = cudaMemset(d_y, 0, num_rows * sizeof(float)); CHECK_CUDA_ERROR(cuda_status); // Initialize output vector to zero

    // SpMV parameters
    float alpha = 1.0f;
    float beta = 0.0f;

    // Perform SpMV calculation using cusparseScsrmv (S=single precision, csr=CSR format, mv=matrix-vector)
    // Warm-up run
    status = cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           num_rows, num_cols, nnz,
                           &alpha, descr,
                           d_values, d_row_offsets, d_col_indices,
                           d_x, &beta, d_y);
    CHECK_CUSPARSE_ERROR(status);
    cudaDeviceSynchronize(); // Wait for warm-up to finish

    // Time the cuSPARSE SpMV operation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    int iterations = 10; // Run multiple times for stable timing
    for (int i = 0; i < iterations; ++i) {
         status = cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               num_rows, num_cols, nnz,
                               &alpha, descr,
                               d_values, d_row_offsets, d_col_indices,
                               d_x, &beta, d_y);
         // No sync inside loop for accurate timing of the sequence
    }
     CHECK_CUSPARSE_ERROR(status); // Check status after loop

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time_ms = 0.0f;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    gpu_time_ms /= iterations; // Average time per iteration

    printf("cuSPARSE GPU Execution Time: %.4f ms\n", gpu_time_ms);

    // Copy result back to host
    cuda_status = cudaMemcpy(h_y_gpu, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost); CHECK_CUDA_ERROR(cuda_status);

    // --- Verification ---
    printf("\nVerifying results...\n");
    bool correct = verifyResults(h_y_cpu, h_y_gpu, num_rows, 1e-4f); // Use slightly looser tolerance for large FP sums
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");

    // --- Performance Comparison ---
    printf("\nPerformance Comparison:\n");
    printf("  CPU Time:        %.4f ms\n", cpu_time_ms);
    printf("  cuSPARSE GPU Time: %.4f ms\n", gpu_time_ms);
    if (gpu_time_ms > 0) { // Avoid division by zero
         printf("  Speedup vs CPU:  %.2fx\n", cpu_time_ms / gpu_time_ms);
    }

    // Calculate GFLOP/s
    double num_operations = 2.0 * nnz; // Multiply-add for each non-zero
    double gpu_gflops = (num_operations / 1e9) / (gpu_time_ms / 1000.0);
    printf("  cuSPARSE GPU Throughput: %.2f GFLOP/s\n", gpu_gflops);


    // --- Cleanup ---
    printf("\nCleaning up...\n");
    // Free device memory
    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    // Destroy cuSPARSE objects
    cusparseDestroyMatDescr(descr);
    cusparseDestroy(handle);

    // Free host memory
    free(h_x);
    free(h_y_cpu);
    free(h_y_gpu);
    freeCSRMatrix(h_matrix);

    // Clean up CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Day 36 Complete.\n");
    return 0;
}
