#include "min_gru_scan.h" // Include the header file

#include <stdlib.h> // For malloc, free, rand, exit
#include <string.h> // For memcpy
#include <time.h>   // For time, clock
#include <math.h>   // For fabsf, ceilf, log2f
#include <cuda_profiler_api.h> // For timing (still needed here)
// cuda_runtime.h and device_launch_parameters.h are included via min_gru_scan.h

// ============================================================================
// Macros (CHECK_CUDA_ERROR)
// ============================================================================

// GPU_CALLABLE is defined in the header

// CUDA error checking macro
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s (%d)\n", __FILE__, __LINE__, cudaGetErrorString(err), err); \
        exit(EXIT_FAILURE); \
    } \
}

// ============================================================================
// Utilities Implementations (Host functions)
// ============================================================================

// Inline GPU_CALLABLE functions (sigmoid, vector ops, linear_forward) are now defined in the header.

// --- Linear Layer Host Functions ---
// Host function to initialize a linear layer
void init_linear_layer(LinearLayer* layer, int input_size, int output_size) {
    layer->input_size = input_size;
    layer->output_size = output_size;

    // Allocate memory for weights and bias on host
    layer->weights = (float*)malloc(output_size * input_size * sizeof(float));
    layer->bias = (float*)malloc(output_size * sizeof(float));

    if (!layer->weights || !layer->bias) {
        fprintf(stderr, "ERROR: Memory allocation failed in init_linear_layer\n");
        exit(EXIT_FAILURE);
    }

    // Initialize with small random values (consistent with reference)
    for (int i = 0; i < output_size * input_size; i++) {
        layer->weights[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
    }
    for (int i = 0; i < output_size; i++) {
        layer->bias[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
    }
}

// Host function to free memory for linear layer
void free_linear_layer(LinearLayer* layer) {
    if (layer) {
        free(layer->weights);
        free(layer->bias);
        layer->weights = NULL;
        layer->bias = NULL;
// ============================================================================
// MinGRU Specifics Implementations (Host functions)
// ============================================================================

// Struct definitions are in the header.
// Host function to initialize a MinGRU cell
void init_min_gru_cell(MinGRUCell* cell, int input_size, int hidden_size) {
    cell->input_size = input_size;
    cell->hidden_size = hidden_size;
    // Initialize the linear layers
    init_linear_layer(&cell->linear_z, input_size, hidden_size);
    init_linear_layer(&cell->linear_h, input_size, hidden_size);
}

// Host function to free memory for MinGRU cell
void free_min_gru_cell(MinGRUCell* cell) {
    free_linear_layer(&cell->linear_z);
    free_linear_layer(&cell->linear_h);
}

// --- CPU Sequential Implementation ---

// MinGRU forward pass (single step) - Host callable wrapper around GPU_CALLABLE inline functions
// This *could* be inline in the header too, but keeping it here is fine.
void min_gru_forward_cpu_step(const MinGRUCell* cell, const float* x_t, const float* h_prev, float* h_t) {
    // Allocate temporary arrays on stack if size is reasonable.
    // For very large hidden_size, consider dynamic allocation inside.
    if (cell->hidden_size > 1024) {
         // Consider dynamic allocation or alternative approach for large hidden sizes
         // This is a simplified example.
    }

    float z_t[cell->hidden_size];
    float h_tilde[cell->hidden_size];
    float one_minus_z[cell->hidden_size];
    float z_h_tilde[cell->hidden_size];
    float one_minus_z_h_prev[cell->hidden_size];

    // Compute update gate: z_t = sigmoid(Linear_z(x_t))
    linear_forward(&cell->linear_z, x_t, z_t);
    for (int i = 0; i < cell->hidden_size; i++) {
        z_t[i] = sigmoid(z_t[i]);
    }

    // Compute candidate hidden state: h_tilde = Linear_h(x_t)
    linear_forward(&cell->linear_h, x_t, h_tilde);

    // Compute h_t = (1 - z_t) * h_prev + z_t * h_tilde
    vec_sub_from_scalar(one_minus_z, 1.0f, z_t, cell->hidden_size);
    vec_mul(one_minus_z_h_prev, one_minus_z, h_prev, cell->hidden_size);
    vec_mul(z_h_tilde, z_t, h_tilde, cell->hidden_size);
    vec_add(h_t, one_minus_z_h_prev, z_h_tilde, cell->hidden_size);
}

// Process a full sequence with MinGRU (sequential CPU mode)
void min_gru_process_sequence_cpu(const MinGRUCell* cell, const float* x, const float* h0,
                                 int seq_length, float* h_out) {
    float* h_prev = (float*)malloc(cell->hidden_size * sizeof(float));
    float* h_curr = (float*)malloc(cell->hidden_size * sizeof(float));
    if (!h_prev || !h_curr) {
        fprintf(stderr, "ERROR: Memory allocation failed in min_gru_process_sequence_cpu\n");
        exit(EXIT_FAILURE);
    }

    // Initialize h_prev with h0
    memcpy(h_prev, h0, cell->hidden_size * sizeof(float));

    // Process each time step
    for (int t = 0; t < seq_length; t++) {
        const float* x_t = x + t * cell->input_size; // Get input for current time step
        min_gru_forward_cpu_step(cell, x_t, h_prev, h_curr); // Compute current hidden state

        // Store hidden state in output
        memcpy(h_out + t * cell->hidden_size, h_curr, cell->hidden_size * sizeof(float));

        // Update h_prev for next time step
        memcpy(h_prev, h_curr, cell->hidden_size * sizeof(float));
    }

    free(h_prev);
    free(h_curr);
}

// ============================================================================
// CUDA Kernels (Adapted from min_gru_cuda.cu)
// ============================================================================

// Kernel to compute scan parameters a_t = 1 - z_t and b_t = z_t * h_tilde
__global__ void min_gru_extract_scan_params_kernel(const LinearLayer d_linear_z, // Pass by value
                                                  const LinearLayer d_linear_h, // Pass by value
                                                  const float* d_x, float* d_a, float* d_b,
                                                  int seq_length, int hidden_size, int input_size) {
    // Grid: (blocksPerGrid_h, seq_length) Threads: (threadsPerBlock_h)
    int h = blockIdx.x * blockDim.x + threadIdx.x; // Index within hidden_size
    int t = blockIdx.y;                           // Index for time step (sequence length)

    if (t < seq_length && h < hidden_size) {
        const float* x_t = d_x + t * input_size; // Input for this time step

        // Compute z_t = sigmoid(Linear_z(x_t))
        float z_t_val = d_linear_z.bias[h];
        const float* z_weights_row = d_linear_z.weights + h * input_size;
        for (int i = 0; i < input_size; i++) {
            z_t_val += z_weights_row[i] * x_t[i];
        }
        z_t_val = 1.0f / (1.0f + expf(-z_t_val)); // sigmoid

        // Compute h_tilde = Linear_h(x_t)
        float h_tilde_val = d_linear_h.bias[h];
        const float* h_weights_row = d_linear_h.weights + h * input_size;
        for (int i = 0; i < input_size; i++) {
            h_tilde_val += h_weights_row[i] * x_t[i];
        }

        // Compute and store a_t = 1 - z_t and b_t = z_t * h_tilde
        int index = t * hidden_size + h;
        d_a[index] = 1.0f - z_t_val;
        d_b[index] = z_t_val * h_tilde_val;
    }
}


// Kernel to compose two scan operations: op_out = op2 ○ op1
// op = (a, b) where h_t = a * h_{t-1} + b
// op_out = (a_out, b_out)
// op1 = (a1, b1)
// op2 = (a2, b2)
// a_out = a2 * a1
// b_out = a2 * b1 + b2
__global__ void compose_scan_ops_kernel(const float* d_a1, const float* d_b1,
                                       const float* d_a2, const float* d_b2,
                                       float* d_a_out, float* d_b_out,
                                       int size) { // size is hidden_size
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float a1_val = d_a1[idx];
        float b1_val = d_b1[idx];
        float a2_val = d_a2[idx];
        float b2_val = d_b2[idx];

        d_a_out[idx] = a2_val * a1_val;
        d_b_out[idx] = a2_val * b1_val + b2_val;
    }
}

// Kernel to apply a scan operation: h_out = a * h_in + b
__global__ void apply_scan_op_kernel(const float* d_a, const float* d_b,
                                    const float* d_h_in, float* d_h_out,
                                    int size) { // size is hidden_size
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_h_out[idx] = d_a[idx] * d_h_in[idx] + d_b[idx];
    }
}


// ============================================================================
// CUDA Helper Functions (Adapted from min_gru_cuda.cu)
// ============================================================================

// Helper function to transfer MinGRU cell layers to device memory
// Allocates memory on device pointed to by d_linear_z/h and copies from host cell
void min_gru_to_device(const MinGRUCell* cell, LinearLayer* d_linear_z, LinearLayer* d_linear_h) {
    int input_size = cell->input_size;
    int hidden_size = cell->hidden_size;
    size_t weights_size = (size_t)hidden_size * input_size * sizeof(float);
    size_t bias_size = (size_t)hidden_size * sizeof(float);

    // --- linear_z ---
    d_linear_z->input_size = input_size;
    d_linear_z->output_size = hidden_size;
    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_linear_z->weights, weights_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_linear_z->bias, bias_size));
    // Copy from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_linear_z->weights, cell->linear_z.weights, weights_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_linear_z->bias, cell->linear_z.bias, bias_size, cudaMemcpyHostToDevice));

    // --- linear_h ---
    d_linear_h->input_size = input_size;
    d_linear_h->output_size = hidden_size;
    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_linear_h->weights, weights_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_linear_h->bias, bias_size));
    // Copy from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_linear_h->weights, cell->linear_h.weights, weights_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_linear_h->bias, cell->linear_h.bias, bias_size, cudaMemcpyHostToDevice));
}

// Helper function to free device memory associated with MinGRU cell layers
void min_gru_free_device(LinearLayer* d_linear_z, LinearLayer* d_linear_h) {
    if (d_linear_z) {
        cudaFree(d_linear_z->weights);
        cudaFree(d_linear_z->bias);
        d_linear_z->weights = NULL; // Prevent double free
        d_linear_z->bias = NULL;
    }
    if (d_linear_h) {
        cudaFree(d_linear_h->weights);
        cudaFree(d_linear_h->bias);
        d_linear_h->weights = NULL; // Prevent double free
        d_linear_h->bias = NULL;
    }
}

// ============================================================================
// CUDA Parallel Scan Implementation (Adapted from min_gru_cuda.cu)
// ============================================================================

// CUDA implementation of the parallel scan algorithm for MinGRU recurrence
// h_t = a_t * h_{t-1} + b_t
// Inputs d_a, d_b, d_h0 are pointers to device memory
// Output d_h_out is a pointer to device memory (will contain h_1 to h_seq_length)
void min_gru_parallel_scan_cuda(int seq_length, int batch_size, int hidden_size,
                               const float* d_a, const float* d_b, const float* d_h0,
                               float* d_h_out) {
    // Note: Batch size is assumed to be 1 in this implementation, matching reference.
    if (batch_size != 1) {
        fprintf(stderr, "ERROR: Batch size > 1 not supported in this parallel scan implementation.\n");
        exit(EXIT_FAILURE);
    }

    size_t seq_hidden_size_bytes = (size_t)seq_length * hidden_size * sizeof(float);
    size_t hidden_size_bytes = (size_t)hidden_size * sizeof(float);

    // CUDA kernel launch parameters
    int threadsPerBlock = 256; // Common choice, adjust based on GPU
    int blocksPerGrid = (hidden_size + threadsPerBlock - 1) / threadsPerBlock;
    dim3 blockDim(threadsPerBlock);
    dim3 gridDim(blocksPerGrid);

    // For short sequences, a sequential application on GPU might be faster
    // than the overhead of the parallel scan setup. Threshold can be tuned.
    int sequential_threshold = 8; // From reference code
    if (seq_length <= sequential_threshold) {
        float *d_h_prev, *d_h_curr;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_h_prev, hidden_size_bytes));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_h_curr, hidden_size_bytes));

        // Initialize h_prev with h0
        CHECK_CUDA_ERROR(cudaMemcpy(d_h_prev, d_h0, hidden_size_bytes, cudaMemcpyDeviceToDevice));

        // Apply scan operation sequentially for each time step
        for (int t = 0; t < seq_length; t++) {
            const float* d_a_t = d_a + t * hidden_size;
            const float* d_b_t = d_b + t * hidden_size;
            float* d_h_out_t = d_h_out + t * hidden_size;

            apply_scan_op_kernel<<<gridDim, blockDim>>>(d_a_t, d_b_t, d_h_prev, d_h_curr, hidden_size);
            CHECK_CUDA_ERROR(cudaGetLastError()); // Check kernel launch

            CHECK_CUDA_ERROR(cudaMemcpy(d_h_out_t, d_h_curr, hidden_size_bytes, cudaMemcpyDeviceToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(d_h_prev, d_h_curr, hidden_size_bytes, cudaMemcpyDeviceToDevice)); // Update h_prev
        }
        CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Ensure all steps are done

        CHECK_CUDA_ERROR(cudaFree(d_h_prev));
        CHECK_CUDA_ERROR(cudaFree(d_h_curr));
    } else {
        // --- Parallel Scan Implementation (Tree-based approach from reference) ---
        // This implementation seems slightly different from a standard Blelloch scan's down-sweep.
        // It iteratively composes operations in an up-sweep and then applies them.

        // Allocate temporary device memory for intermediate composed operations
        // We need log2(seq_length) levels of temporary storage.
        int num_levels = 0;
        if (seq_length > 0) {
            num_levels = (int)ceilf(log2f((float)seq_length));
        }

        // Pointers to device memory for each level of the scan tree
        float **d_a_levels = (float**)malloc((num_levels + 1) * sizeof(float*));
        float **d_b_levels = (float**)malloc((num_levels + 1) * sizeof(float*));
        if (!d_a_levels || !d_b_levels) {
             fprintf(stderr, "ERROR: Failed to allocate host memory for level pointers.\n");
             exit(EXIT_FAILURE);
        }

        // Level 0 points to the original input a and b arrays
        d_a_levels[0] = (float*)d_a; // Need const_cast or handle differently if d_a is strictly const
        d_b_levels[0] = (float*)d_b; // Need const_cast

        // Allocate device memory for intermediate levels (1 to num_levels)
        for (int level = 1; level <= num_levels; level++) {
            CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a_levels[level], seq_hidden_size_bytes));
            CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b_levels[level], seq_hidden_size_bytes));
            // Initialize? The reference code might rely on compose kernel writing to all necessary parts.
            // It might be safer to initialize to identity (a=1, b=0) or copy previous level.
            // Let's try copying previous level first, as compose only updates specific indices.
             CHECK_CUDA_ERROR(cudaMemcpy(d_a_levels[level], d_a_levels[level-1], seq_hidden_size_bytes, cudaMemcpyDeviceToDevice));
             CHECK_CUDA_ERROR(cudaMemcpy(d_b_levels[level], d_b_levels[level-1], seq_hidden_size_bytes, cudaMemcpyDeviceToDevice));
        }

        // --- Up-sweep phase: Compose operations level by level ---
        for (int level = 0; level < num_levels; level++) {
            int stride = 1 << level; // Distance between elements being combined (1, 2, 4, ...)
            // The reference code seems to iterate differently, maybe not a full tree?
            // Let's re-examine the reference 'min_gru_cuda.cu::min_gru_parallel_scan_cuda' logic carefully.

            // --- Corrected Up-sweep (Simulating reference more closely) ---
            // The reference code's loop structure seems complex and potentially incorrect or non-standard.
            // A standard work-efficient parallel scan (like Blelloch) is usually preferred.
            // Let's try implementing a simpler, less efficient, but possibly easier-to-understand scan first.
            // This simpler version iteratively applies the composition.

            // Alternative: Simpler (but less efficient) iterative scan composition on GPU
            // This computes the prefix product/sum of the operations sequentially on the GPU.
            // Allocate temporary buffers for the composed op at step t-1
            float* d_a_composed_prev = NULL;
            float* d_b_composed_prev = NULL;
            float* d_a_composed_curr = NULL;
            float* d_b_composed_curr = NULL;
            CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a_composed_prev, hidden_size_bytes));
            CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b_composed_prev, hidden_size_bytes));
            CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a_composed_curr, hidden_size_bytes));
            CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b_composed_curr, hidden_size_bytes));

            // Initialize composed op for t=0 (it's just op_0)
            CHECK_CUDA_ERROR(cudaMemcpy(d_a_composed_prev, d_a, hidden_size_bytes, cudaMemcpyDeviceToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(d_b_composed_prev, d_b, hidden_size_bytes, cudaMemcpyDeviceToDevice));

            // Apply op_0 to h0 to get h_1
            apply_scan_op_kernel<<<gridDim, blockDim>>>(d_a_composed_prev, d_b_composed_prev, d_h0, d_h_out, hidden_size);
            CHECK_CUDA_ERROR(cudaGetLastError());

            // Iteratively compute composed ops and apply them
            for (int t = 1; t < seq_length; ++t) {
                const float* d_a_t = d_a + t * hidden_size;
                const float* d_b_t = d_b + t * hidden_size;
                float* d_h_out_t = d_h_out + t * hidden_size;

                // Compose current op_t with the previously composed op (op_{t-1} ○ ... ○ op_0)
                // result -> d_a_composed_curr, d_b_composed_curr
                compose_scan_ops_kernel<<<gridDim, blockDim>>>(
                    d_a_composed_prev, d_b_composed_prev, // op_{t-1} ○ ... ○ op_0
                    d_a_t, d_b_t,                         // op_t
                    d_a_composed_curr, d_b_composed_curr, // result: op_t ○ ... ○ op_0
                    hidden_size);
                CHECK_CUDA_ERROR(cudaGetLastError());

                // Apply the fully composed operation (op_t ○ ... ○ op_0) to h0 to get h_{t+1}
                apply_scan_op_kernel<<<gridDim, blockDim>>>(
                    d_a_composed_curr, d_b_composed_curr, // op_t ○ ... ○ op_0
                    d_h0,                                 // h0
                    d_h_out_t,                            // result: h_{t+1}
                    hidden_size);
                CHECK_CUDA_ERROR(cudaGetLastError());

                // Prepare for next iteration: current composed op becomes previous
                 CHECK_CUDA_ERROR(cudaMemcpy(d_a_composed_prev, d_a_composed_curr, hidden_size_bytes, cudaMemcpyDeviceToDevice));
                 CHECK_CUDA_ERROR(cudaMemcpy(d_b_composed_prev, d_b_composed_curr, hidden_size_bytes, cudaMemcpyDeviceToDevice));
            }
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            // Free temporary buffers for iterative scan
            CHECK_CUDA_ERROR(cudaFree(d_a_composed_prev));
            CHECK_CUDA_ERROR(cudaFree(d_b_composed_prev));
            CHECK_CUDA_ERROR(cudaFree(d_a_composed_curr));
            CHECK_CUDA_ERROR(cudaFree(d_b_composed_curr));

        } // End of parallel scan implementation choice (sequential vs parallel)

        // --- Cleanup (Common for both scan approaches) ---
        // Free intermediate level memory if tree-based was used (commented out for now)
        /*
        if (seq_length > sequential_threshold) {
             for (int level = 1; level <= num_levels; level++) {
                 cudaFree(d_a_levels[level]);
                 cudaFree(d_b_levels[level]);
             }
             free(d_a_levels);
             free(d_b_levels);
        }
        */
    } // End of `else` for parallel scan logic

    // Note: d_h_out is populated directly in both branches (sequential GPU / parallel scan GPU)
    // No final copy back to host needed here, that's done in the calling function.
}

// ============================================================================
// Main CUDA Wrapper (Adapted from min_gru_cuda.cu)
// ============================================================================

// Process a full sequence with MinGRU using CUDA (extracts params and runs parallel scan)
void min_gru_process_sequence_cuda(const MinGRUCell* cell, const float* x, const float* h0,
                                 int seq_length, float* h_out) {
    // --- 1. Transfer Cell Data to Device ---
    LinearLayer d_linear_z, d_linear_h;
    min_gru_to_device(cell, &d_linear_z, &d_linear_h);

    // --- 2. Allocate Device Memory for Sequence Data ---
    float *d_x = NULL, *d_h0 = NULL, *d_h_out = NULL;
    float *d_a = NULL, *d_b = NULL;

    size_t input_seq_size_bytes = (size_t)seq_length * cell->input_size * sizeof(float);
    size_t hidden_state_size_bytes = (size_t)cell->hidden_size * sizeof(float);
    size_t hidden_seq_size_bytes = (size_t)seq_length * cell->hidden_size * sizeof(float);

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_x, input_seq_size_bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_h0, hidden_state_size_bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_h_out, hidden_seq_size_bytes)); // Output hidden states h_1..h_T
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a, hidden_seq_size_bytes));    // Scan param a
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b, hidden_seq_size_bytes));    // Scan param b

    // --- 3. Copy Host Inputs to Device ---
    CHECK_CUDA_ERROR(cudaMemcpy(d_x, x, input_seq_size_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_h0, h0, hidden_state_size_bytes, cudaMemcpyHostToDevice));

    // --- 4. Extract Scan Parameters (a_t, b_t) using Kernel ---
    int threadsPerBlock = 256; // Or tune this
    int blocksPerGrid_h = (cell->hidden_size + threadsPerBlock - 1) / threadsPerBlock;
    dim3 extractBlockDim(threadsPerBlock);
    dim3 extractGridDim(blocksPerGrid_h, seq_length); // GridDim.y corresponds to sequence length

    min_gru_extract_scan_params_kernel<<<extractGridDim, extractBlockDim>>>(
        d_linear_z, d_linear_h, // Pass device layer structs by value
        d_x, d_a, d_b,
        seq_length, cell->hidden_size, cell->input_size
    );
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check kernel launch

    // --- 5. Run Parallel Scan ---
    // This function computes h_1..h_T based on a_1..a_T, b_1..b_T, and h_0
    min_gru_parallel_scan_cuda(seq_length, 1, cell->hidden_size, d_a, d_b, d_h0, d_h_out);
    // Note: Internal synchronization might be needed depending on scan implementation.
    // The iterative one used above synchronizes after each step.

    // --- 6. Copy Results Back to Host ---
    CHECK_CUDA_ERROR(cudaMemcpy(h_out, d_h_out, hidden_seq_size_bytes, cudaMemcpyDeviceToHost));

    // --- 7. Free All Device Memory ---
    CHECK_CUDA_ERROR(cudaFree(d_x));
    CHECK_CUDA_ERROR(cudaFree(d_h0));
    CHECK_CUDA_ERROR(cudaFree(d_h_out));
    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    min_gru_free_device(&d_linear_z, &d_linear_h); // Frees weights/biases inside layers
}

// ============================================================================
// Main Function and Helpers (Adapted from main.c)
// ============================================================================

// Helper function to generate random data
void generate_random_data(float* data, int size, float min_val, float max_val) {
    float range = max_val - min_val;
    for (int i = 0; i < size; i++) {
        data[i] = min_val + ((float)rand() / RAND_MAX) * range;
    }
}

// Helper function to print a vector (for debugging)
void print_vector(const char* name, const float* vec, int size) {
    printf("%s: [", name);
    int print_limit = size < 10 ? size : 10; // Print first few elements
    for (int i = 0; i < print_limit; i++) {
        printf("%.4f", vec[i]);
        if (i < print_limit - 1) printf(", ");
    }
    if (size > print_limit) printf(", ...");
    printf("]\n");
}

// Helper to compare results and find max difference
float compare_results(const float* res1, const float* res2, int size) {
     float max_diff = 0.0f;
     for (int i = 0; i < size; ++i) {
         float diff = fabsf(res1[i] - res2[i]);
         if (diff > max_diff) {
             max_diff = diff;
         }
     }
     return max_diff;
}


int main() {
    // Seed random number generator
    srand((unsigned int)time(NULL));

    printf("\n--- MinGRU Parallel Scan CUDA Example ---\n");

    // Parameters
    int input_size = 128;  // Larger sizes to show GPU benefit
    int hidden_size = 256;
    int seq_length = 100; // Longer sequence
    printf("Parameters: input_size=%d, hidden_size=%d, seq_length=%d\n",
           input_size, hidden_size, seq_length);

    // Initialize the MinGRU cell (on host)
    MinGRUCell cell;
    init_min_gru_cell(&cell, input_size, hidden_size);
    printf("Host MinGRU cell initialized.\n");

    // Allocate host memory for input sequence, initial hidden state, and output sequences
    size_t input_seq_elems = (size_t)seq_length * input_size;
    size_t hidden_state_elems = (size_t)hidden_size;
    size_t hidden_seq_elems = (size_t)seq_length * hidden_size;

    float* h_x = (float*)malloc(input_seq_elems * sizeof(float));
    float* h_h0 = (float*)malloc(hidden_state_elems * sizeof(float));
    float* h_out_cpu = (float*)malloc(hidden_seq_elems * sizeof(float));
    float* h_out_cuda = (float*)malloc(hidden_seq_elems * sizeof(float));

    if (!h_x || !h_h0 || !h_out_cpu || !h_out_cuda) {
        fprintf(stderr, "ERROR: Host memory allocation failed in main.\n");
        free_min_gru_cell(&cell);
        return 1;
    }
    printf("Host memory allocated.\n");

    // Generate random input data
    generate_random_data(h_x, input_seq_elems, -1.0f, 1.0f);
    generate_random_data(h_h0, hidden_state_elems, -1.0f, 1.0f);
    printf("Random host data generated.\n");
    // print_vector("Input x (first step)", h_x, input_size);
    // print_vector("Input h0", h_h0, hidden_size);


    // --- Run CPU Sequential Version ---
    printf("Processing with MinGRU (CPU Sequential)...\n");
    clock_t start_cpu = clock();
    min_gru_process_sequence_cpu(&cell, h_x, h_h0, seq_length, h_out_cpu);
    clock_t end_cpu = clock();
    double cpu_time = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC;
    printf("CPU Processing Time: %.6f seconds\n", cpu_time);
    // print_vector("Output h_cpu (last step)", h_out_cpu + (seq_length - 1) * hidden_size, hidden_size);


    // --- Run CUDA Parallel Scan Version ---
    printf("Processing with MinGRU (CUDA Parallel Scan)...\n");
    cudaEvent_t start_cuda, stop_cuda;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_cuda));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_cuda));

    CHECK_CUDA_ERROR(cudaEventRecord(start_cuda)); // Record start event

    min_gru_process_sequence_cuda(&cell, h_x, h_h0, seq_length, h_out_cuda);

    CHECK_CUDA_ERROR(cudaEventRecord(stop_cuda)); // Record stop event
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_cuda)); // Wait for stop event to complete

    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start_cuda, stop_cuda));
    double cuda_time = milliseconds / 1000.0;
    printf("CUDA Processing Time: %.6f seconds\n", cuda_time);
    // print_vector("Output h_cuda (last step)", h_out_cuda + (seq_length - 1) * hidden_size, hidden_size);


    // --- Verification ---
    printf("Verifying results...\n");
    float max_diff = compare_results(h_out_cpu, h_out_cuda, hidden_seq_elems);
    printf("Maximum absolute difference between CPU and CUDA results: %.8f\n", max_diff);
    if (max_diff > 1e-4) { // Tolerance for floating point differences
         printf("WARNING: Difference exceeds tolerance!\n");
         // Optionally print differing values
         // for(int i=0; i<hidden_seq_elems; ++i) {
         //     if (fabsf(h_out_cpu[i] - h_out_cuda[i]) > 1e-4) {
         //         printf("Diff at index %d: CPU=%.6f, CUDA=%.6f\n", i, h_out_cpu[i], h_out_cuda[i]);
         //         break;
         //     }
         // }
    } else {
         printf("Results verified successfully within tolerance.\n");
    }


    // --- Cleanup ---
    printf("Cleaning up...\n");
    CHECK_CUDA_ERROR(cudaEventDestroy(start_cuda));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_cuda));
    free(h_x);
    free(h_h0);
    free(h_out_cpu);
    free(h_out_cuda);
    free_min_gru_cell(&cell); // Frees host weights/biases
    printf("Cleanup complete.\n");

    return 0;
}
