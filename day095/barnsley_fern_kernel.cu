#include "barnsley_fern.cuh"
#include <curand_kernel.h>

// Affine transformation parameters for Barnsley Fern
// These could be in __constant__ memory for slight performance gain if they don't change.
// For simplicity, defined here. Note: probabilities are cumulative for selection.

// f1: Stem
const float f1_a = 0.00f; const float f1_b = 0.00f; const float f1_c = 0.00f; const float f1_d = 0.16f; const float f1_e = 0.00f; const float f1_f = 0.00f;
const float p1 = 0.01f;

// f2: Successively smaller leaflets
const float f2_a = 0.85f; const float f2_b = 0.04f; const float f2_c = -0.04f; const float f2_d = 0.85f; const float f2_e = 0.00f; const float f2_f = 1.60f;
const float p2 = 0.01f + 0.85f; // Cumulative probability

// f3: Largest left leaflet
const float f3_a = 0.20f; const float f3_b = -0.26f; const float f3_c = 0.23f; const float f3_d = 0.22f; const float f3_e = 0.00f; const float f3_f = 1.60f;
const float p3 = 0.01f + 0.85f + 0.07f; // Cumulative probability

// f4: Largest right leaflet
const float f4_a = -0.15f; const float f4_b = 0.28f; const float f4_c = 0.26f; const float f4_d = 0.24f; const float f4_e = 0.00f; const float f4_f = 0.44f;
// p4 is implicitly up to 1.0

__global__ void setup_kernel(curandState_t *state, unsigned long long seed, int num_threads) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_threads) {
        curand_init(seed + id, id, 0, &state[id]);
    }
}

__global__ void generate_fern_kernel(
    unsigned int* image_buffer,
    int image_width,
    int image_height,
    int num_iterations_per_thread, // Renamed for clarity
    curandState_t* rand_states,
    float fern_x_min, float fern_x_max, float fern_y_min, float fern_y_max,
    int warmup_iterations) {

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curandState_t local_rand_state = rand_states[id];

    float x = 0.0f;
    float y = 0.0f;
    float next_x, next_y;

    for (int i = 0; i < num_iterations_per_thread + warmup_iterations; ++i) {
        float r = curand_uniform(&local_rand_state);

        if (r < p1) {
            next_x = f1_a * x + f1_b * y + f1_e;
            next_y = f1_c * x + f1_d * y + f1_f;
        } else if (r < p2) {
            next_x = f2_a * x + f2_b * y + f2_e;
            next_y = f2_c * x + f2_d * y + f2_f;
        } else if (r < p3) {
            next_x = f3_a * x + f3_b * y + f3_e;
            next_y = f3_c * x + f3_d * y + f3_f;
        } else {
            next_x = f4_a * x + f4_b * y + f4_e;
            next_y = f4_c * x + f4_d * y + f4_f;
        }
        x = next_x;
        y = next_y;

        if (i >= warmup_iterations) {
            // Map fractal coordinates to image pixel coordinates
            // Ensure that fern_x_max > fern_x_min and fern_y_max > fern_y_min to avoid division by zero or negative scaling
            int px = static_cast<int>(((x - fern_x_min) / (fern_x_max - fern_x_min)) * image_width);
            int py = static_cast<int>(((fern_y_max - y) / (fern_y_max - fern_y_min)) * image_height); // Invert y-axis for typical image coordinates (0,0 at top-left)

            if (px >= 0 && px < image_width && py >= 0 && py < image_height) {
                atomicAdd(&image_buffer[py * image_width + px], 1);
            }
        }
    }
    rand_states[id] = local_rand_state; // Save state back if needed, though for this app, maybe not critical after generation
}
