#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>

// Error checking macro
#define cudaCheckError() {\
    cudaError_t e = cudaGetLastError();\
    if (e != cudaSuccess) {\
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));\
        exit(EXIT_FAILURE);\
    }\
}

// CSR format sparse matrix structure
typedef struct {
    int num_rows;       // Number of rows
    int num_cols;       // Number of columns
    int num_nonzeros;   // Number of non-zero elements
    int *row_offsets;   // Row offsets array (size: num_rows + 1)
    int *col_indices;   // Column indices array (size: num_nonzeros)
    float *values;      // Values array (size: num_nonzeros)
} CSRMatrix;

/**
 * CUDA kernel for Sparse Matrix-Vector Multiplication (SpMV)
 * Each thread computes one output element y[i]
 */
__global__ void spmvCSRKernel(
    const int num_rows,
    const int *row_offsets,
    const int *col_indices,
    const float *values,
    const float *x,
    float *y
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < num_rows) {
        float dot = 0.0f;
        int row_start = row_offsets[row];
        int row_end = row_offsets[row + 1];
        
        // Compute dot product for this row
        for (int i = row_start; i < row_end; i++) {
            dot += values[i] * x[col_indices[i]];
        }
        
        y[row] = dot;
    }
}

/**
 * Alternative CUDA kernel for SpMV with shared memory optimization
 * This version uses shared memory to cache the input vector x
 */
__global__ void spmvCSRKernelOptimized(
    const int num_rows,
    const int num_cols,
    const int *row_offsets,
    const int *col_indices,
    const float *values,
    const float *x,
    float *y
) {
    extern __shared__ float x_shared[];
    
    // Collaboratively load x into shared memory
    for (int i = threadIdx.x; i < num_cols; i += blockDim.x) {
        x_shared[i] = x[i];
    }
    
    __syncthreads();
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < num_rows) {
        float dot = 0.0f;
        int row_start = row_offsets[row];
        int row_end = row_offsets[row + 1];
        
        // Compute dot product for this row using cached x values
        for (int i = row_start; i < row_end; i++) {
            dot += values[i] * x_shared[col_indices[i]];
        }
        
        y[row] = dot;
    }
}

/**
 * CPU implementation of SpMV for verification
 */
void spmvCPU(
    const CSRMatrix *matrix,
    const float *x,
    float *y
) {
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

/**
 * Generate a random sparse matrix with specified dimensions and sparsity
 */
CSRMatrix* generateRandomSparseMatrix(int num_rows, int num_cols, float sparsity) {
    CSRMatrix *matrix = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    matrix->num_rows = num_rows;
    matrix->num_cols = num_cols;
    
    // Estimate number of non-zeros based on sparsity
    int max_nonzeros = num_rows * num_cols;
    int estimated_nonzeros = (int)(max_nonzeros * sparsity);
    
    // Allocate arrays with some extra space
    matrix->row_offsets = (int*)malloc((num_rows + 1) * sizeof(int));
    matrix->col_indices = (int*)malloc(estimated_nonzeros * sizeof(int));
    matrix->values = (float*)malloc(estimated_nonzeros * sizeof(float));
    
    // Initialize row_offsets
    matrix->row_offsets[0] = 0;
    int nnz = 0;
    
    // Generate random sparse matrix
    srand(time(NULL));
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            float r = (float)rand() / RAND_MAX;
            if (r < sparsity) {
                if (nnz < estimated_nonzeros) {
                    matrix->col_indices[nnz] = j;
                    matrix->values[nnz] = (float)rand() / RAND_MAX * 10.0f;
                    nnz++;
                }
            }
        }
        matrix->row_offsets[i + 1] = nnz;
    }
    
    // Update actual number of non-zeros
    matrix->num_nonzeros = nnz;
    
    // Ensure each row has at least one non-zero element
    for (int i = 0; i < num_rows; i++) {
        if (matrix->row_offsets[i] == matrix->row_offsets[i + 1]) {
            // This row has no non-zero elements, add one
            int col = rand() % num_cols;
            
            // Shift all elements after this row
            for (int j = num_rows; j > i; j--) {
                matrix->row_offsets[j]++;
            }
            
            // Make space for the new element
            for (int j = matrix->num_nonzeros; j > matrix->row_offsets[i]; j--) {
                matrix->col_indices[j] = matrix->col_indices[j - 1];
                matrix->values[j] = matrix->values[j - 1];
            }
            
            // Insert the new element
            matrix->col_indices[matrix->row_offsets[i]] = col;
            matrix->values[matrix->row_offsets[i]] = (float)rand() / RAND_MAX * 10.0f;
            matrix->num_nonzeros++;
        }
    }
    
    return matrix;
}

/**
 * Free memory allocated for CSR matrix
 */
void freeCSRMatrix(CSRMatrix *matrix) {
    if (matrix) {
        free(matrix->row_offsets);
        free(matrix->col_indices);
        free(matrix->values);
        free(matrix);
    }
}

/**
 * Print matrix statistics and a sample of its contents
 */
void printMatrixInfo(const CSRMatrix *matrix) {
    printf("Sparse Matrix Info:\n");
    printf("  Dimensions: %d x %d\n", matrix->num_rows, matrix->num_cols);
    printf("  Non-zeros: %d\n", matrix->num_nonzeros);
    printf("  Sparsity: %.4f%%\n", 100.0f * matrix->num_nonzeros / (matrix->num_rows * matrix->num_cols));
    
    // Print a small sample of the matrix
    int sample_size = matrix->num_rows < 5 ? matrix->num_rows : 5;
    printf("\nSample of matrix (first %d rows):\n", sample_size);
    
    for (int i = 0; i < sample_size; i++) {
        printf("Row %d: ", i);
        for (int j = matrix->row_offsets[i]; j < matrix->row_offsets[i + 1]; j++) {
            printf("(%d, %.2f) ", matrix->col_indices[j], matrix->values[j]);
        }
        printf("\n");
    }
}

/**
 * Verify results by comparing CPU and GPU outputs
 */
bool verifyResults(const float *cpu_result, const float *gpu_result, int size, float tolerance) {
    for (int i = 0; i < size; i++) {
        float diff = fabs(cpu_result[i] - gpu_result[i]);
        float rel_diff = diff / (fabs(cpu_result[i]) + 1e-6f);
        if (rel_diff > tolerance) {
            printf("Verification failed at index %d: CPU = %f, GPU = %f\n", i, cpu_result[i], gpu_result[i]);
            return false;
        }
    }
    return true;
}

/**
 * Main function to demonstrate SpMV operations
 */
int main(int argc, char **argv) {
    // Set default parameters
    int num_rows = 10000;
    int num_cols = 10000;
    float sparsity = 0.01f;  // 1% non-zero elements
    
    // Parse command line arguments if provided
    if (argc > 1) num_rows = atoi(argv[1]);
    if (argc > 2) num_cols = atoi(argv[2]);
    if (argc > 3) sparsity = atof(argv[3]);
    
    printf("Sparse Matrix-Vector Multiplication (SpMV)\n");
    printf("Matrix size: %d x %d, Sparsity: %.2f%%\n\n", num_rows, num_cols, sparsity * 100);
    
    // Generate random sparse matrix
    CSRMatrix *matrix = generateRandomSparseMatrix(num_rows, num_cols, sparsity);
    printMatrixInfo(matrix);
    
    // Generate random input vector
    float *h_x = (float*)malloc(num_cols * sizeof(float));
    for (int i = 0; i < num_cols; i++) {
        h_x[i] = (float)rand() / RAND_MAX * 10.0f;
    }
    
    // Allocate memory for output vectors
    float *h_y_cpu = (float*)malloc(num_rows * sizeof(float));
    float *h_y_gpu = (float*)malloc(num_rows * sizeof(float));
    float *h_y_gpu_optimized = (float*)malloc(num_rows * sizeof(float));
    
    // Compute reference result on CPU
    clock_t cpu_start = clock();
    spmvCPU(matrix, h_x, h_y_cpu);
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0;  // ms
    
    // Allocate device memory
    int *d_row_offsets, *d_col_indices;
    float *d_values, *d_x, *d_y;
    
    cudaMalloc((void**)&d_row_offsets, (matrix->num_rows + 1) * sizeof(int));
    cudaMalloc((void**)&d_col_indices, matrix->num_nonzeros * sizeof(int));
    cudaMalloc((void**)&d_values, matrix->num_nonzeros * sizeof(float));
    cudaMalloc((void**)&d_x, num_cols * sizeof(float));
    cudaMalloc((void**)&d_y, num_rows * sizeof(float));
    cudaCheckError();
    
    // Copy data to device
    cudaMemcpy(d_row_offsets, matrix->row_offsets, (matrix->num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, matrix->col_indices, matrix->num_nonzeros * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, matrix->values, matrix->num_nonzeros * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, num_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError();
    
    // Set kernel parameters
    int block_size = 256;
    int grid_size = (num_rows + block_size - 1) / block_size;
    
    // Warm-up run
    spmvCSRKernel<<<grid_size, block_size>>>(num_rows, d_row_offsets, d_col_indices, d_values, d_x, d_y);
    cudaDeviceSynchronize();
    cudaCheckError();
    
    // Measure basic kernel performance
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {  // Run multiple times for more accurate timing
        spmvCSRKernel<<<grid_size, block_size>>>(num_rows, d_row_offsets, d_col_indices, d_values, d_x, d_y);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_time_ms = 0.0f;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    gpu_time_ms /= 10.0f;  // Average time per run
    
    // Copy results back to host
    cudaMemcpy(h_y_gpu, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();
    
    // Run optimized kernel with shared memory
    size_t shared_mem_size = num_cols * sizeof(float);
    
    // Check if we have enough shared memory
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    if (shared_mem_size <= prop.sharedMemPerBlock) {
        // Warm-up run
        spmvCSRKernelOptimized<<<grid_size, block_size, shared_mem_size>>>(num_rows, num_cols, d_row_offsets, d_col_indices, d_values, d_x, d_y);
        cudaDeviceSynchronize();
        cudaCheckError();
        
        // Measure optimized kernel performance
        cudaEventRecord(start);
        for (int i = 0; i < 10; i++) {
            spmvCSRKernelOptimized<<<grid_size, block_size, shared_mem_size>>>(num_rows, num_cols, d_row_offsets, d_col_indices, d_values, d_x, d_y);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float gpu_optimized_time_ms = 0.0f;
        cudaEventElapsedTime(&gpu_optimized_time_ms, start, stop);
        gpu_optimized_time_ms /= 10.0f;  // Average time per run
        
        // Copy results back to host
        cudaMemcpy(h_y_gpu_optimized, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost);
        cudaCheckError();
        
        // Verify optimized kernel results
        bool optimized_correct = verifyResults(h_y_cpu, h_y_gpu_optimized, num_rows, 1e-5f);
        
        printf("\nOptimized GPU SpMV (Shared Memory): %s\n", optimized_correct ? "PASSED" : "FAILED");
        printf("  Execution time: %.4f ms\n", gpu_optimized_time_ms);
        printf("  Speedup vs CPU: %.2fx\n", cpu_time / gpu_optimized_time_ms);
        printf("  Speedup vs Basic GPU: %.2fx\n", gpu_time_ms / gpu_optimized_time_ms);
    } else {
        printf("\nSkipping optimized kernel: Not enough shared memory (need %zu bytes, have %zu bytes)\n", 
               shared_mem_size, prop.sharedMemPerBlock);
    }
    
    // Verify basic kernel results
    bool basic_correct = verifyResults(h_y_cpu, h_y_gpu, num_rows, 1e-5f);
    
    // Print performance results
    printf("\nPerformance Results:\n");
    printf("CPU SpMV: %.4f ms\n", cpu_time);
    printf("Basic GPU SpMV: %s\n", basic_correct ? "PASSED" : "FAILED");
    printf("  Execution time: %.4f ms\n", gpu_time_ms);
    printf("  Speedup vs CPU: %.2fx\n", cpu_time / gpu_time_ms);
    
    // Calculate throughput
    double num_operations = 2.0 * matrix->num_nonzeros;  // Each non-zero element requires a multiply and add
    double gpu_throughput = (num_operations / 1e9) / (gpu_time_ms / 1000.0);  // GFLOP/s
    
    printf("\nBasic GPU SpMV Throughput: %.2f GFLOP/s\n", gpu_throughput);
    
    // Clean up
    free(h_x);
    free(h_y_cpu);
    free(h_y_gpu);
    free(h_y_gpu_optimized);
    freeCSRMatrix(matrix);
    
    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
