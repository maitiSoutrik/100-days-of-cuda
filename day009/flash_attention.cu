#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// Error checking macro
#define cudaCheckError() {\
    cudaError_t e = cudaGetLastError();\
    if (e != cudaSuccess) {\
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));\
        exit(EXIT_FAILURE);\
    }\
}

// Constants for FlashAttention
#define BLOCK_SIZE 16
#define MAX_SEQ_LENGTH 512
#define EPSILON 1e-6f

/**
 * Flash Attention Forward Pass Kernel
 * 
 * This is a simplified implementation of the Flash Attention algorithm
 * described in the paper "FlashAttention: Fast and Memory-Efficient Exact 
 * Attention with IO-Awareness" by Dao et al.
 * 
 * The key idea is to compute attention in blocks to reduce memory bandwidth
 * and improve efficiency.
 */
__global__ void flashAttentionForward(
    const float* Q,       // Query matrix [batch_size, seq_len, head_dim]
    const float* K,       // Key matrix [batch_size, seq_len, head_dim]
    const float* V,       // Value matrix [batch_size, seq_len, head_dim]
    float* O,             // Output matrix [batch_size, seq_len, head_dim]
    const int batch_size,
    const int seq_len,
    const int head_dim,
    const float scale     // Scaling factor (1/sqrt(head_dim))
) {
    // Shared memory for block-wise computation
    extern __shared__ float shared_mem[];
    
    // Divide shared memory into blocks for Q, K, V, and intermediate results
    float* Q_block = shared_mem;
    float* K_block = Q_block + BLOCK_SIZE * head_dim;
    float* V_block = K_block + BLOCK_SIZE * head_dim;
    float* S_block = V_block + BLOCK_SIZE * head_dim;  // For storing attention scores
    
    // Thread indices
    const int batch_idx = blockIdx.z;
    const int row_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int col_idx = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    
    // Local variables for accumulating results
    float m_i = -INFINITY;  // Max value for numerical stability
    float l_i = 0.0f;       // Sum of exponentials
    float y_i[16] = {0.0f}; // Output accumulator (assuming head_dim <= 16 for simplicity)
    
    // Check if this thread is within bounds
    const bool valid_row = row_idx < seq_len;
    
    // Process blocks of K and V
    for (int block_start = 0; block_start < seq_len; block_start += BLOCK_SIZE) {
        // Load Q block into shared memory (only once per row block)
        if (valid_row && threadIdx.y < head_dim && block_start == blockIdx.y * BLOCK_SIZE) {
            Q_block[threadIdx.x * head_dim + threadIdx.y] = 
                Q[(batch_idx * seq_len + row_idx) * head_dim + threadIdx.y];
        }
        
        // Load K and V blocks into shared memory
        const int k_col = block_start + threadIdx.y;
        if (valid_row && k_col < seq_len && threadIdx.y < head_dim) {
            K_block[threadIdx.x * head_dim + threadIdx.y] = 
                K[(batch_idx * seq_len + k_col) * head_dim + threadIdx.x];
            V_block[threadIdx.x * head_dim + threadIdx.y] = 
                V[(batch_idx * seq_len + k_col) * head_dim + threadIdx.x];
        }
        
        __syncthreads();
        
        // Compute attention scores for this block
        if (valid_row && block_start + threadIdx.y < seq_len) {
            float s_ij = 0.0f;
            
            // Compute dot product of Q_i and K_j
            for (int h = 0; h < head_dim; h++) {
                s_ij += Q_block[threadIdx.x * head_dim + h] * K_block[threadIdx.y * head_dim + h];
            }
            
            // Apply scaling
            s_ij *= scale;
            
            // Store in shared memory
            S_block[threadIdx.x * BLOCK_SIZE + threadIdx.y] = s_ij;
            
            // Update max value for numerical stability
            m_i = fmaxf(m_i, s_ij);
        }
        
        __syncthreads();
        
        // Compute softmax and output for this block
        if (valid_row) {
            // Recompute with updated m_i for numerical stability
            float m_i_prev = m_i;
            float l_i_prev = l_i;
            
            // Update accumulators
            for (int j = 0; j < BLOCK_SIZE && block_start + j < seq_len; j++) {
                float s_ij = S_block[threadIdx.x * BLOCK_SIZE + j];
                float p_ij = expf(s_ij - m_i_prev);
                
                // Update output accumulator
                for (int h = 0; h < head_dim; h++) {
                    y_i[h] += p_ij * V_block[j * head_dim + h];
                }
                
                l_i += p_ij;
            }
            
            // Rescale previous outputs if m_i changed
            if (m_i_prev != m_i && l_i_prev > 0) {
                float scale_factor = expf(m_i_prev - m_i);
                l_i = l_i_prev * scale_factor + l_i;
                
                for (int h = 0; h < head_dim; h++) {
                    y_i[h] *= scale_factor;
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write final output
    if (valid_row) {
        for (int h = 0; h < head_dim; h++) {
            O[(batch_idx * seq_len + row_idx) * head_dim + h] = y_i[h] / (l_i + EPSILON);
        }
    }
}

/**
 * Host function to perform Flash Attention forward pass
 */
void computeFlashAttention(
    float* h_Q, float* h_K, float* h_V, float* h_O,
    int batch_size, int seq_len, int head_dim
) {
    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O;
    size_t matrix_size = batch_size * seq_len * head_dim * sizeof(float);
    
    cudaMalloc((void**)&d_Q, matrix_size);
    cudaMalloc((void**)&d_K, matrix_size);
    cudaMalloc((void**)&d_V, matrix_size);
    cudaMalloc((void**)&d_O, matrix_size);
    cudaCheckError();
    
    // Copy input data to device
    cudaMemcpy(d_Q, h_Q, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, matrix_size, cudaMemcpyHostToDevice);
    cudaCheckError();
    
    // Calculate grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(
        (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE,
        batch_size
    );
    
    // Calculate shared memory size
    size_t shared_mem_size = (3 * BLOCK_SIZE * head_dim + BLOCK_SIZE * BLOCK_SIZE) * sizeof(float);
    
    // Compute scaling factor
    float scale = 1.0f / sqrtf(head_dim);
    
    // Launch kernel
    flashAttentionForward<<<gridDim, blockDim, shared_mem_size>>>(
        d_Q, d_K, d_V, d_O, batch_size, seq_len, head_dim, scale
    );
    cudaCheckError();
    
    // Copy results back to host
    cudaMemcpy(h_O, d_O, matrix_size, cudaMemcpyDeviceToHost);
    cudaCheckError();
    
    // Free device memory
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaCheckError();
}

/**
 * CPU implementation of attention for verification
 */
void cpuAttention(
    float* Q, float* K, float* V, float* O,
    int batch_size, int seq_len, int head_dim
) {
    float scale = 1.0f / sqrtf(head_dim);
    
    for (int b = 0; b < batch_size; b++) {
        // Compute attention scores for each sequence position
        for (int i = 0; i < seq_len; i++) {
            float max_val = -INFINITY;
            
            // Temporary storage for attention scores
            float* scores = (float*)malloc(seq_len * sizeof(float));
            
            // Compute attention scores and find max for numerical stability
            for (int j = 0; j < seq_len; j++) {
                float dot_product = 0.0f;
                for (int h = 0; h < head_dim; h++) {
                    dot_product += Q[(b * seq_len + i) * head_dim + h] * K[(b * seq_len + j) * head_dim + h];
                }
                scores[j] = dot_product * scale;
                max_val = fmaxf(max_val, scores[j]);
            }
            
            // Compute softmax denominator
            float sum_exp = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                scores[j] = expf(scores[j] - max_val);
                sum_exp += scores[j];
            }
            
            // Compute weighted sum of values
            for (int h = 0; h < head_dim; h++) {
                float weighted_sum = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    weighted_sum += scores[j] * V[(b * seq_len + j) * head_dim + h];
                }
                O[(b * seq_len + i) * head_dim + h] = weighted_sum / sum_exp;
            }
            
            free(scores);
        }
    }
}

/**
 * Initialize matrices with random values
 */
void initializeRandomData(float* matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f; // Values between -1 and 1
    }
}

/**
 * Check if results match within tolerance
 */
bool compareResults(float* A, float* B, int size, float tolerance) {
    for (int i = 0; i < size; i++) {
        if (fabsf(A[i] - B[i]) > tolerance) {
            printf("Mismatch at index %d: CPU = %f, GPU = %f\n", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

int main() {
    // Set parameters for the attention mechanism
    const int batch_size = 2;
    const int seq_len = 64;   // Keep this small for demonstration
    const int head_dim = 8;   // Keep this small for demonstration
    
    // Allocate host memory
    size_t matrix_size = batch_size * seq_len * head_dim;
    size_t matrix_bytes = matrix_size * sizeof(float);
    
    float* h_Q = (float*)malloc(matrix_bytes);
    float* h_K = (float*)malloc(matrix_bytes);
    float* h_V = (float*)malloc(matrix_bytes);
    float* h_O_gpu = (float*)malloc(matrix_bytes);
    float* h_O_cpu = (float*)malloc(matrix_bytes);
    
    // Initialize input matrices with random data
    srand(42); // For reproducibility
    initializeRandomData(h_Q, matrix_size);
    initializeRandomData(h_K, matrix_size);
    initializeRandomData(h_V, matrix_size);
    
    // Compute attention using GPU (Flash Attention)
    printf("Computing Flash Attention on GPU...\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    computeFlashAttention(h_Q, h_K, h_V, h_O_gpu, batch_size, seq_len, head_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("GPU Time: %.3f ms\n", gpu_time);
    
    // Compute attention using CPU for verification
    printf("Computing Attention on CPU for verification...\n");
    clock_t cpu_start = clock();
    cpuAttention(h_Q, h_K, h_V, h_O_cpu, batch_size, seq_len, head_dim);
    clock_t cpu_end = clock();
    float cpu_time = 1000.0f * (float)(cpu_end - cpu_start) / CLOCKS_PER_SEC;
    printf("CPU Time: %.3f ms\n", cpu_time);
    
    // Compare results
    printf("Comparing results...\n");
    bool match = compareResults(h_O_cpu, h_O_gpu, matrix_size, 1e-2f);
    if (match) {
        printf("Results match within tolerance!\n");
    } else {
        printf("Results do not match!\n");
    }
    
    // Print speedup
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    
    // Demonstrate attention computation with a small example
    printf("\nDemonstrating Attention with a small example:\n");
    const int demo_size = 4;
    const int demo_dim = 2;
    
    // Create small example matrices
    float demo_Q[demo_size * demo_dim] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    float demo_K[demo_size * demo_dim] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    float demo_V[demo_size * demo_dim] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float demo_O[demo_size * demo_dim] = {0.0f};
    
    // Print input matrices
    printf("Q matrix:\n");
    for (int i = 0; i < demo_size; i++) {
        for (int j = 0; j < demo_dim; j++) {
            printf("%.1f ", demo_Q[i * demo_dim + j]);
        }
        printf("\n");
    }
    
    printf("\nK matrix:\n");
    for (int i = 0; i < demo_size; i++) {
        for (int j = 0; j < demo_dim; j++) {
            printf("%.1f ", demo_K[i * demo_dim + j]);
        }
        printf("\n");
    }
    
    printf("\nV matrix:\n");
    for (int i = 0; i < demo_size; i++) {
        for (int j = 0; j < demo_dim; j++) {
            printf("%.1f ", demo_V[i * demo_dim + j]);
        }
        printf("\n");
    }
    
    // Compute attention for demo
    cpuAttention(demo_Q, demo_K, demo_V, demo_O, 1, demo_size, demo_dim);
    
    // Print attention matrix and result
    printf("\nAttention Scores (Q * K^T / sqrt(dim)):\n");
    float scale_demo = 1.0f / sqrtf(demo_dim);
    for (int i = 0; i < demo_size; i++) {
        for (int j = 0; j < demo_size; j++) {
            float dot_product = 0.0f;
            for (int h = 0; h < demo_dim; h++) {
                dot_product += demo_Q[i * demo_dim + h] * demo_K[j * demo_dim + h];
            }
            printf("%.3f ", dot_product * scale_demo);
        }
        printf("\n");
    }
    
    printf("\nSoftmax(Attention Scores):\n");
    for (int i = 0; i < demo_size; i++) {
        // Compute attention scores
        float scores[demo_size];
        float max_val = -INFINITY;
        
        for (int j = 0; j < demo_size; j++) {
            float dot_product = 0.0f;
            for (int h = 0; h < demo_dim; h++) {
                dot_product += demo_Q[i * demo_dim + h] * demo_K[j * demo_dim + h];
            }
            scores[j] = dot_product * scale_demo;
            max_val = fmaxf(max_val, scores[j]);
        }
        
        // Apply softmax
        float sum_exp = 0.0f;
        for (int j = 0; j < demo_size; j++) {
            scores[j] = expf(scores[j] - max_val);
            sum_exp += scores[j];
        }
        
        // Print normalized scores
        for (int j = 0; j < demo_size; j++) {
            printf("%.3f ", scores[j] / sum_exp);
        }
        printf("\n");
    }
    
    printf("\nOutput (O = Softmax(QK^T/sqrt(d)) * V):\n");
    for (int i = 0; i < demo_size; i++) {
        for (int j = 0; j < demo_dim; j++) {
            printf("%.3f ", demo_O[i * demo_dim + j]);
        }
        printf("\n");
    }
    
    // Free memory
    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_O_gpu);
    free(h_O_cpu);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
