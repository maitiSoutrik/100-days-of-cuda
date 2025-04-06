#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// --- CUDA Error Checking Macro ---
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

// --- Constants ---
#define BLOCK_SIZE 256

// --- Kernels ---

// Kernel to calculate gradients for each sample in the mini-batch
__global__ void calculate_gradients_kernel(
    const float* d_x_batch,
    const float* d_y_true_batch,
    const float* d_w,
    const float* d_b,
    float* d_gradients_w,
    float* d_gradients_b,
    int batch_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < batch_size) {
        // Load data for this sample
        float x = d_x_batch[tid];
        float y_true = d_y_true_batch[tid];
        
        // Load current model parameters
        float w = *d_w;
        float b = *d_b;
        
        // Calculate prediction
        float y_pred = w * x + b;
        
        // Calculate error
        float error = y_pred - y_true;
        
        // Calculate gradients
        d_gradients_w[tid] = 2.0f * error * x;
        d_gradients_b[tid] = 2.0f * error;
    }
}

// Kernel to reduce (sum) gradients across the mini-batch
__global__ void reduce_sum_kernel(
    const float* d_input,
    float* d_output,
    int n
) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < n) ? d_input[i] : 0.0f;
    __syncthreads();
    
    // Perform parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        d_output[blockIdx.x] = sdata[0];
    }
}

// --- Host Functions ---

// Function to generate synthetic training data
void generate_data(float* x, float* y, int n, float true_w, float true_b, float noise_scale) {
    srand(time(NULL));
    
    for (int i = 0; i < n; i++) {
        // Generate x values between -10 and 10
        x[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;
        
        // Generate y = true_w * x + true_b + noise
        float noise = ((float)rand() / RAND_MAX) * 2.0f * noise_scale - noise_scale;
        y[i] = true_w * x[i] + true_b + noise;
    }
}

// Function to calculate MSE loss
float calculate_mse(const float* x, const float* y_true, float w, float b, int n) {
    float mse = 0.0f;
    
    for (int i = 0; i < n; i++) {
        float y_pred = w * x[i] + b;
        float error = y_pred - y_true[i];
        mse += error * error;
    }
    
    return mse / n;
}

// Function to shuffle data (Fisher-Yates algorithm)
void shuffle_data(float* x, float* y, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        
        // Swap x[i] and x[j]
        float temp_x = x[i];
        x[i] = x[j];
        x[j] = temp_x;
        
        // Swap y[i] and y[j]
        float temp_y = y[i];
        y[i] = y[j];
        y[j] = temp_y;
    }
}

// --- Main Function ---
int main() {
    // --- Parameters ---
    const int data_size = 10000;
    const int batch_size = 128;
    const int num_epochs = 50;
    const float learning_rate = 0.01f;
    
    // True parameters (to be learned)
    const float true_w = 2.5f;
    const float true_b = 1.2f;
    const float noise_scale = 0.5f;
    
    printf("Mini-Batch SGD for Linear Regression\n");
    printf("-------------------------------------\n");
    printf("Data size: %d\n", data_size);
    printf("Batch size: %d\n", batch_size);
    printf("Number of epochs: %d\n", num_epochs);
    printf("Learning rate: %.4f\n", learning_rate);
    printf("True parameters: w = %.2f, b = %.2f\n", true_w, true_b);
    printf("Noise scale: %.2f\n", noise_scale);
    printf("-------------------------------------\n\n");
    
    // --- Host Memory Allocation ---
    
    // Allocate memory for full dataset
    float* h_x_full = (float*)malloc(data_size * sizeof(float));
    float* h_y_true_full = (float*)malloc(data_size * sizeof(float));
    
    if (!h_x_full || !h_y_true_full) {
        fprintf(stderr, "Failed to allocate host memory for dataset\n");
        return EXIT_FAILURE;
    }
    
    // Generate synthetic data
    generate_data(h_x_full, h_y_true_full, data_size, true_w, true_b, noise_scale);
    
    // Allocate memory for a single mini-batch
    float* h_x_batch = (float*)malloc(batch_size * sizeof(float));
    float* h_y_true_batch = (float*)malloc(batch_size * sizeof(float));
    
    if (!h_x_batch || !h_y_true_batch) {
        fprintf(stderr, "Failed to allocate host memory for mini-batch\n");
        return EXIT_FAILURE;
    }
    
    // Initialize model parameters
    float h_w = 0.0f;  // Initial weight
    float h_b = 0.0f;  // Initial bias
    
    // Allocate memory for gradient sums
    float h_sum_gradient_w = 0.0f;
    float h_sum_gradient_b = 0.0f;
    
    // --- Device Memory Allocation ---
    
    // Allocate memory for mini-batch
    float *d_x_batch, *d_y_true_batch;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_x_batch, batch_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_y_true_batch, batch_size * sizeof(float)));
    
    // Allocate memory for model parameters
    float *d_w, *d_b;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_w, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b, sizeof(float)));
    
    // Allocate memory for per-sample gradients
    float *d_gradients_w, *d_gradients_b;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_gradients_w, batch_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_gradients_b, batch_size * sizeof(float)));
    
    // Allocate memory for gradient sums
    float *d_sum_gradient_w, *d_sum_gradient_b;
    
    // Calculate number of blocks needed for reduction
    int num_blocks = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_sum_gradient_w, num_blocks * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_sum_gradient_b, num_blocks * sizeof(float)));
    
    // Allocate memory for final reduction result if multiple blocks
    float *d_final_sum_w = NULL, *d_final_sum_b = NULL;
    if (num_blocks > 1) {
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_final_sum_w, sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_final_sum_b, sizeof(float)));
    }
    
    // --- Training Loop ---
    printf("Starting training...\n");
    
    // Calculate initial loss
    float initial_loss = calculate_mse(h_x_full, h_y_true_full, h_w, h_b, data_size);
    printf("Initial loss: %.6f\n", initial_loss);
    
    // Configure kernel launch parameters
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Reduction grid for final sum (if needed)
    dim3 finalGridDim(1);
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Shuffle data at the beginning of each epoch
        shuffle_data(h_x_full, h_y_true_full, data_size);
        
        // Mini-batch loop
        for (int batch_start = 0; batch_start < data_size; batch_start += batch_size) {
            // Adjust batch size for the last batch if needed
            int current_batch_size = batch_size;
            if (batch_start + batch_size > data_size) {
                current_batch_size = data_size - batch_start;
            }
            
            // Copy current batch to host buffers
            for (int i = 0; i < current_batch_size; i++) {
                h_x_batch[i] = h_x_full[batch_start + i];
                h_y_true_batch[i] = h_y_true_full[batch_start + i];
            }
            
            // Copy mini-batch data to device
            CHECK_CUDA_ERROR(cudaMemcpy(d_x_batch, h_x_batch, current_batch_size * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(d_y_true_batch, h_y_true_batch, current_batch_size * sizeof(float), cudaMemcpyHostToDevice));
            
            // Copy current model parameters to device
            CHECK_CUDA_ERROR(cudaMemcpy(d_w, &h_w, sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(d_b, &h_b, sizeof(float), cudaMemcpyHostToDevice));
            
            // Calculate gradients for each sample in the mini-batch
            calculate_gradients_kernel<<<gridDim, blockDim>>>(
                d_x_batch, d_y_true_batch, d_w, d_b, d_gradients_w, d_gradients_b, current_batch_size
            );
            CHECK_CUDA_ERROR(cudaGetLastError());
            
            // Reduce (sum) gradients across the mini-batch
            reduce_sum_kernel<<<num_blocks, blockDim>>>(d_gradients_w, d_sum_gradient_w, current_batch_size);
            CHECK_CUDA_ERROR(cudaGetLastError());
            
            reduce_sum_kernel<<<num_blocks, blockDim>>>(d_gradients_b, d_sum_gradient_b, current_batch_size);
            CHECK_CUDA_ERROR(cudaGetLastError());
            
            // If multiple blocks were used, perform a final reduction
            if (num_blocks > 1) {
                reduce_sum_kernel<<<finalGridDim, blockDim>>>(d_sum_gradient_w, d_final_sum_w, num_blocks);
                CHECK_CUDA_ERROR(cudaGetLastError());
                
                reduce_sum_kernel<<<finalGridDim, blockDim>>>(d_sum_gradient_b, d_final_sum_b, num_blocks);
                CHECK_CUDA_ERROR(cudaGetLastError());
                
                // Copy final sums back to host
                CHECK_CUDA_ERROR(cudaMemcpy(&h_sum_gradient_w, d_final_sum_w, sizeof(float), cudaMemcpyDeviceToHost));
                CHECK_CUDA_ERROR(cudaMemcpy(&h_sum_gradient_b, d_final_sum_b, sizeof(float), cudaMemcpyDeviceToHost));
            } else {
                // Copy sums back to host
                CHECK_CUDA_ERROR(cudaMemcpy(&h_sum_gradient_w, d_sum_gradient_w, sizeof(float), cudaMemcpyDeviceToHost));
                CHECK_CUDA_ERROR(cudaMemcpy(&h_sum_gradient_b, d_sum_gradient_b, sizeof(float), cudaMemcpyDeviceToHost));
            }
            
            // Update model parameters on host
            float avg_gradient_w = h_sum_gradient_w / current_batch_size;
            float avg_gradient_b = h_sum_gradient_b / current_batch_size;
            
            h_w = h_w - learning_rate * avg_gradient_w;
            h_b = h_b - learning_rate * avg_gradient_b;
        }
        
        // Calculate and print loss after each epoch
        if ((epoch + 1) % 5 == 0 || epoch == 0) {
            float current_loss = calculate_mse(h_x_full, h_y_true_full, h_w, h_b, data_size);
            printf("Epoch %d/%d - Loss: %.6f, w: %.4f, b: %.4f\n", 
                   epoch + 1, num_epochs, current_loss, h_w, h_b);
        }
    }
    
    // --- Final Results ---
    float final_loss = calculate_mse(h_x_full, h_y_true_full, h_w, h_b, data_size);
    
    printf("\nTraining completed!\n");
    printf("-------------------------------------\n");
    printf("Initial parameters: w = %.4f, b = %.4f\n", 0.0f, 0.0f);
    printf("Learned parameters: w = %.4f, b = %.4f\n", h_w, h_b);
    printf("True parameters:    w = %.4f, b = %.4f\n", true_w, true_b);
    printf("Final loss: %.6f\n", final_loss);
    printf("-------------------------------------\n");
    
    // Calculate relative error
    float w_error = fabs((h_w - true_w) / true_w) * 100.0f;
    float b_error = fabs((h_b - true_b) / true_b) * 100.0f;
    printf("Relative error: w = %.2f%%, b = %.2f%%\n", w_error, b_error);
    
    // --- Cleanup ---
    
    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_x_batch));
    CHECK_CUDA_ERROR(cudaFree(d_y_true_batch));
    CHECK_CUDA_ERROR(cudaFree(d_w));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_gradients_w));
    CHECK_CUDA_ERROR(cudaFree(d_gradients_b));
    CHECK_CUDA_ERROR(cudaFree(d_sum_gradient_w));
    CHECK_CUDA_ERROR(cudaFree(d_sum_gradient_b));
    
    if (num_blocks > 1) {
        CHECK_CUDA_ERROR(cudaFree(d_final_sum_w));
        CHECK_CUDA_ERROR(cudaFree(d_final_sum_b));
    }
    
    // Free host memory
    free(h_x_full);
    free(h_y_true_full);
    free(h_x_batch);
    free(h_y_true_batch);
    
    printf("\nDay 28 Mini-Batch SGD finished successfully.\n");
    return EXIT_SUCCESS;
}
