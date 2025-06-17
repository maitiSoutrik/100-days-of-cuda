#include "mandelbrot_amr.cuh"
#include <iostream> // For std::cout, std::cerr
#include <vector>   // For std::vector (host-side region management if needed)
#include <cmath>    // For fabs, etc.

// Device function to calculate Mandelbrot iteration count
__device__ int mandelbrot_iterations(double cx, double cy, int max_iterations) {
    double x = 0.0;
    double y = 0.0;
    int iterations = 0;
    while (x * x + y * y <= 4.0 && iterations < max_iterations) {
        double xtemp = x * x - y * y + cx;
        y = 2.0 * x * y + cy;
        x = xtemp;
        iterations++;
    }
    return iterations;
}

// Kernel to compute Mandelbrot set for a given region and trigger sub-launches
__global__ void mandelbrot_kernel(unsigned char* image_data,
                                  int global_width, int global_height,
                                  Region current_region,
                                  int max_iterations,
                                  int max_depth,
                                  float refinement_threshold,
                                  int* sub_regions_count_dev) {
    int px = blockIdx.x * blockDim.x + threadIdx.x; // Pixel x in current_region
    int py = blockIdx.y * blockDim.y + threadIdx.y; // Pixel y in current_region

    if (px >= current_region.width_px || py >= current_region.height_px) {
        return;
    }

    // Map pixel coordinates to complex plane
    double cx = current_region.x_min + (current_region.x_max - current_region.x_min) * px / (current_region.width_px -1);
    double cy = current_region.y_min + (current_region.y_max - current_region.y_min) * py / (current_region.height_px -1);

    int iterations = mandelbrot_iterations(cx, cy, max_iterations);
    unsigned char color = (unsigned char)(255 * (iterations / (float)max_iterations));
    if (iterations == max_iterations) {
        color = 0; // Typically black for points inside the set
    }
    
    // Write to global image buffer
    int global_px = current_region.start_x_global + px;
    int global_py = current_region.start_y_global + py;
    if (global_px < global_width && global_py < global_height) {
         image_data[global_py * global_width + global_px] = color;
    }


    // Adaptive Mesh Refinement (AMR) Logic
    // This is a simplified check. A more robust check would analyze variance in a small patch.
    // For this example, we'll consider subdividing if we are not at max depth and the point is "interesting"
    // (e.g., not trivially inside or outside, or if a block contains mixed values).
    // This part needs to be carefully designed.
    // A common strategy is to have a separate kernel check for refinement needs after a region is computed.
    // Or, each block can vote. For simplicity, let's assume a block decides to refine if it contains boundary points.

    // This kernel's primary job is to compute the pixels.
    // Refinement decision and sub-launching will be coordinated.
    // Let's assume for now that if a thread is on a boundary (iterations < max_iterations but > 0),
    // it might indicate a complex region.
    // A more robust approach: after this kernel computes a region, another small kernel
    // checks the computed pixel data for this region. If variance is high, then subdivide.

    // For dynamic parallelism, a thread (typically one per block, or a designated thread)
    // can launch new kernels.
    if (current_region.current_depth < max_depth) {
        // Simplified condition for refinement: if this thread is part of a block
        // that might need refinement. This is tricky to coordinate efficiently.
        // A better approach is to have a "refinement check" kernel after this one.
        // However, to demonstrate dynamic launch from *this* kernel:
        // Let's say if a thread finds a point near the boundary, it tries to launch.
        // This needs careful synchronization to avoid multiple launches for the same sub-region.

        // Example: If this is the first thread in a block, and some condition is met
        // (e.g., based on shared memory analysis of the block's results), launch sub-kernels.
        // This is a placeholder for a more complex refinement check.
        // For now, we'll handle refinement launching in the host wrapper or a dedicated parent kernel.
        // The `sub_regions_count_dev` could be used by a block to signal it found an area to refine.
        // atomicAdd(sub_regions_count_dev, 1); // if this block thinks its area needs refinement
    }
}


// Host function to manage the AMR process
void generate_mandelbrot_amr(unsigned char* image_data_host, int global_width, int global_height,
                             double initial_x_min, double initial_x_max,
                             double initial_y_min, double initial_y_max,
                             int max_iterations, int max_depth, float refinement_threshold) {

    unsigned char* image_data_dev;
    CHECK_CUDA_ERROR(cudaMalloc(&image_data_dev, global_width * global_height * sizeof(unsigned char)));
    CHECK_CUDA_ERROR(cudaMemset(image_data_dev, 255, global_width * global_height * sizeof(unsigned char))); // Initialize to white

    // For dynamic parallelism, we need to manage a queue of regions to process.
    // This can be done with a std::vector on host, or a more complex device-side queue.
    std::vector<Region> regions_to_process;

    Region initial_region;
    initial_region.x_min = initial_x_min; initial_region.x_max = initial_x_max;
    initial_region.y_min = initial_y_min; initial_region.y_max = initial_y_max;
    initial_region.width_px = global_width; initial_region.height_px = global_height;
    initial_region.start_x_global = 0; initial_region.start_y_global = 0;
    initial_region.current_depth = 0;
    regions_to_process.push_back(initial_region);

    dim3 block_dim(16, 16); // Example block dimension

    // This loop simulates a queue of regions. In a full dynamic parallelism scenario,
    // kernels themselves would add to a device-side queue.
    // For simplicity here, the host manages the queue based on results.
    int head = 0;
    while(head < regions_to_process.size()){
        Region current_region = regions_to_process[head++];
        
        if (current_region.current_depth >= max_depth) {
            // If max depth reached, just compute this region without further subdivision
            dim3 grid_dim((current_region.width_px + block_dim.x - 1) / block_dim.x,
                          (current_region.height_px + block_dim.y - 1) / block_dim.y);
            
            // No sub_regions_count_dev needed if not subdividing further from this call
            mandelbrot_kernel<<<grid_dim, block_dim>>>(image_data_dev, global_width, global_height,
                                                  current_region, max_iterations, max_depth,
                                                  refinement_threshold, nullptr);
            CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for kernel to complete
            continue;
        }

        // --- Stage 1: Compute current region ---
        dim3 grid_dim((current_region.width_px + block_dim.x - 1) / block_dim.x,
                      (current_region.height_px + block_dim.y - 1) / block_dim.y);
        
        mandelbrot_kernel<<<grid_dim, block_dim>>>(image_data_dev, global_width, global_height,
                                              current_region, max_iterations, max_depth,
                                              refinement_threshold, nullptr); // Simplified: not using sub_regions_count_dev here
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Ensure this region is computed before checking for refinement

        // --- Stage 2: Check for refinement (Simplified host-side check) ---
        // A more advanced version would use a kernel to check for refinement.
        // Here, we simulate this by checking if we are below max_depth and then subdivide.
        // The actual "need" for refinement would be based on pixel variance in the region.
        // For this example, we will always subdivide if current_depth < max_depth.
        // A true AMR would use `refinement_threshold`.

        // Let's assume we always subdivide into 4 sub-regions if not at max_depth
        if (current_region.current_depth < max_depth) {
            double mid_x = (current_region.x_min + current_region.x_max) / 2.0;
            double mid_y = (current_region.y_min + current_region.y_max) / 2.0;
            int mid_px_w = current_region.width_px / 2;
            int mid_px_h = current_region.height_px / 2;

            Region sub_regions[4];
            // Top-left
            sub_regions[0] = {current_region.x_min, mid_x, current_region.y_min, mid_y,
                              mid_px_w, mid_px_h,
                              current_region.start_x_global, current_region.start_y_global,
                              current_region.current_depth + 1};
            // Top-right
            sub_regions[1] = {mid_x, current_region.x_max, current_region.y_min, mid_y,
                              current_region.width_px - mid_px_w, mid_px_h,
                              current_region.start_x_global + mid_px_w, current_region.start_y_global,
                              current_region.current_depth + 1};
            // Bottom-left
            sub_regions[2] = {current_region.x_min, mid_x, mid_y, current_region.y_max,
                              mid_px_w, current_region.height_px - mid_px_h,
                              current_region.start_x_global, current_region.start_y_global + mid_px_h,
                              current_region.current_depth + 1};
            // Bottom-right
            sub_regions[3] = {mid_x, current_region.x_max, mid_y, current_region.y_max,
                              current_region.width_px - mid_px_w, current_region.height_px - mid_px_h,
                              current_region.start_x_global + mid_px_w, current_region.start_y_global + mid_px_h,
                              current_region.current_depth + 1};

            for (int i = 0; i < 4; ++i) {
                if (sub_regions[i].width_px > 0 && sub_regions[i].height_px > 0) {
                     // In a true dynamic parallelism setup from device, the mandelbrot_kernel itself
                     // would launch new mandelbrot_kernels for these sub_regions.
                     // Here, the host adds them to the queue.
                    regions_to_process.push_back(sub_regions[i]);
                }
            }
        }
    }


    CHECK_CUDA_ERROR(cudaMemcpy(image_data_host, image_data_dev, global_width * global_height * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaFree(image_data_dev));
}

// The check_refinement_kernel is not fully implemented here as the host-side controls refinement for simplicity.
// A full device-side AMR would implement this to analyze a computed region and decide on subdivision.
__global__ void check_refinement_kernel(const unsigned char* region_pixel_data,
                                        int region_width_px, int region_height_px,
                                        float refinement_threshold,
                                        bool* needs_refinement_flag_dev) {
    // This kernel would be launched after a region is computed by mandelbrot_kernel.
    // It would read the pixel values for that region (passed in region_pixel_data,
    // which would be a temporary buffer holding just that region's pixels).
    // It would then calculate variance or some other metric.
    // If variance > refinement_threshold, it sets *needs_refinement_flag_dev = true.
    // Only one thread (e.g., threadIdx.x == 0 && blockIdx.x == 0) would write to needs_refinement_flag_dev.
    // This is a placeholder.
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        // Example: check corners and center for significant differences
        // This is a very naive check.
        unsigned char c00 = region_pixel_data[0];
        unsigned char c01 = region_pixel_data[region_width_px -1];
        unsigned char c10 = region_pixel_data[(region_height_px-1) * region_width_px];
        unsigned char c11 = region_pixel_data[(region_height_px-1) * region_width_px + (region_width_px-1)];
        unsigned char center = region_pixel_data[(region_height_px/2) * region_width_px + (region_width_px/2)];

        float avg = (c00 + c01 + c10 + c11 + center) / 5.0f;
        float variance = (fabs(c00 - avg) + fabs(c01 - avg) + fabs(c10 - avg) + fabs(c11 - avg) + fabs(center - avg)) / 5.0f;

        if (variance > refinement_threshold * 255.0f) { // Assuming threshold is 0-1
            *needs_refinement_flag_dev = true;
        } else {
            *needs_refinement_flag_dev = false;
        }
    }
}
