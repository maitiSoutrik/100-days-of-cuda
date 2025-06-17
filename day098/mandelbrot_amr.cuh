#ifndef MANDELBROT_AMR_CUH
#define MANDELBROT_AMR_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h> // For threadIdx, blockIdx, etc.
#include <cstdio> // For printf in kernels (debugging)

// Error checking macro
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(err), cudaGetErrorName(err), func);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

// Structure to define a region for computation
typedef struct {
    double x_min, x_max;
    double y_min, y_max;
    int width_px, height_px; // Pixels in this region
    int start_x_global, start_y_global; // Top-left corner in global image pixels
    int current_depth;
} Region;

// Function to launch the Mandelbrot generation with adaptive refinement
void generate_mandelbrot_amr(unsigned char* image_data, int global_width, int global_height,
                             double initial_x_min, double initial_x_max,
                             double initial_y_min, double initial_y_max,
                             int max_iterations, int max_depth, float refinement_threshold);

// Device function to calculate Mandelbrot iteration count for a point
__device__ int mandelbrot_iterations(double cx, double cy, int max_iterations);

// Kernel to compute Mandelbrot set for a given region
__global__ void mandelbrot_kernel(unsigned char* image_data,
                                  int global_width, int global_height,
                                  Region current_region,
                                  int max_iterations,
                                  int max_depth,
                                  float refinement_threshold,
                                  int* sub_regions_count_dev); // For counting sub-regions needing refinement

// Kernel to check if a region needs refinement
__global__ void check_refinement_kernel(const unsigned char* region_pixel_data,
                                        int region_width_px, int region_height_px,
                                        float refinement_threshold,
                                        bool* needs_refinement_flag_dev);

#endif // MANDELBROT_AMR_CUH
