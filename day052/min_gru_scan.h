#ifndef MIN_GRU_SCAN_H
#define MIN_GRU_SCAN_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h> // For size_t, FILE etc.

// ============================================================================
// Macros & Forward Declarations
// ============================================================================

#ifdef __HIP_DEVICE_COMPILE__
#define GPU_CALLABLE __host__ __device__
#else
#define GPU_CALLABLE __host__ __device__
#endif

// Forward declare CHECK_CUDA_ERROR for use in headers if needed,
// though its definition usually stays in .cu files.
// Alternatively, include a common error header if you have one.

// ============================================================================
// Utilities (Declarations)
// ============================================================================

// --- Linear Layer ---
typedef struct {
    float* weights;  // [output_size x input_size] (Row-major)
    float* bias;     // [output_size]
    int input_size;
    int output_size;
} LinearLayer;

// Host function declarations for Linear Layer
void init_linear_layer(LinearLayer* layer, int input_size, int output_size);
void free_linear_layer(LinearLayer* layer);

// Device/Host callable forward pass (defined inline or in .cu)
GPU_CALLABLE inline void linear_forward(const LinearLayer* layer, const float* input, float* output); // Definition needed if called from test directly

// --- Vector Ops (can be inline in header or defined in .cu) ---
// Declare if defined in .cu, or define inline here if simple
GPU_CALLABLE inline float sigmoid(float x);
GPU_CALLABLE inline void vec_mul(float* result, const float* a, const float* b, int size);
GPU_CALLABLE inline void vec_add(float* result, const float* a, const float* b, int size);
// ... other vec ops declarations if needed ...

// ============================================================================
// MinGRU Specifics (Declarations)
// ============================================================================

typedef struct {
    LinearLayer linear_z;
    LinearLayer linear_h;
    int hidden_size;
    int input_size;
} MinGRUCell;

// Host function declarations
void init_min_gru_cell(MinGRUCell* cell, int input_size, int hidden_size);
void free_min_gru_cell(MinGRUCell* cell);

// CPU Sequential Implementation (Declaration)
void min_gru_process_sequence_cpu(const MinGRUCell* cell, const float* x, const float* h0,
                                 int seq_length, float* h_out);

// CUDA Implementation (Declaration)
void min_gru_process_sequence_cuda(const MinGRUCell* cell, const float* x, const float* h0,
                                 int seq_length, float* h_out);

// Helper function declarations (if needed by tests)
void generate_random_data(float* data, int size, float min_val, float max_val);
float compare_results(const float* res1, const float* res2, int size);

// ============================================================================
// Inline Definitions for GPU_CALLABLE functions (needed for separate compilation)
// ============================================================================

// Sigmoid activation function
GPU_CALLABLE inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// --- Vector Operations ---
GPU_CALLABLE inline void vec_mul(float* result, const float* a, const float* b, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] * b[i];
    }
}

GPU_CALLABLE inline void vec_add(float* result, const float* a, const float* b, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] + b[i];
    }
}

GPU_CALLABLE inline void vec_scale(float* result, const float* a, float scalar, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] * scalar;
    }
}

GPU_CALLABLE inline void vec_sub_from_scalar(float* result, float scalar, const float* a, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = scalar - a[i];
    }
}

GPU_CALLABLE inline void vec_div_elementwise(float* result, const float* a, const float* b, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] / (b[i] + 1e-8f); // Add epsilon for numerical stability
    }
}

// Forward pass for linear layer: output = weights * input + bias
GPU_CALLABLE inline void linear_forward(const LinearLayer* layer, const float* input, float* output) {
    for (int o = 0; o < layer->output_size; o++) {
        output[o] = layer->bias[o];
        const float* weight_row = layer->weights + o * layer->input_size;
        for (int i = 0; i < layer->input_size; i++) {
            output[o] += weight_row[i] * input[i];
        }
    }
}


#endif // MIN_GRU_SCAN_H
