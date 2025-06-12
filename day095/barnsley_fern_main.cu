#include "barnsley_fern.cuh"
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm> // For std::min_element, std::max_element
#include <cmath>     // For std::fabs, std::round

// Forward declaration of the kernel launcher if not in .cuh
// void generate_fern_kernel_launcher(...);

void save_pgm(const char* filename, const unsigned int* buffer, int width, int height);

int main(int argc, char** argv) {
    std::cout << "Day 095: Barnsley Fern Fractal Generator" << std::endl;

    // Parameters
    const int image_width = 1000;
    const int image_height = 1000;
    const int num_total_points = 200 * 1000 * 1000; // 200 million points
    const int threads_per_block = 256;
    const int num_blocks = (num_total_points / threads_per_block) / 1000; // Adjust points per thread later
                                                                      // Let's aim for roughly 1000 points per thread initially
    const int num_points_per_thread = (num_total_points + (threads_per_block * num_blocks) -1) / (threads_per_block * num_blocks);

    // Fern affine transformation parameters (will be passed to kernel or defined as constants there)
    // Define the bounding box for the fractal
    const float x_min_fern = -2.1820f;
    const float x_max_fern = 2.6558f;
    const float y_min_fern = 0.0f;
    const float y_max_fern = 9.9983f;

    std::cout << "Image dimensions: " << image_width << "x" << image_height << std::endl;
    std::cout << "Total points to generate: " << num_total_points << std::endl;
    std::cout << "Threads per block: " << threads_per_block << std::endl;
    std::cout << "Number of blocks: " << num_blocks << std::endl;
    std::cout << "Points per thread: " << num_points_per_thread << std::endl;

    // Host image buffer
    std::vector<unsigned int> h_image_buffer(image_width * image_height, 0);

    // Device image buffer
    unsigned int* d_image_buffer = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&d_image_buffer, image_width * image_height * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMemset(d_image_buffer, 0, image_width * image_height * sizeof(unsigned int)));

    // cuRAND states
    curandState* d_rand_states = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&d_rand_states, num_blocks * threads_per_block * sizeof(curandState)));

    // Initialize cuRAND states
    unsigned long long seed = time(0); // Use current time as seed
    int total_threads = num_blocks * threads_per_block;
    // Kernel to setup random states
    // Forward declaration for setup_kernel (if not in .cuh, which it is not for now)
    // If setup_kernel is in the same .cu file and defined before its call, this is fine.
    // Otherwise, a declaration would be needed or move its definition before main or into .cuh
    extern __global__ void setup_kernel(curandState *state, unsigned long long seed, int num_threads);
    setup_kernel<<<num_blocks, threads_per_block>>>(d_rand_states, seed, total_threads);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    std::cout << "cuRAND states initialized." << std::endl;

    // Launch the generate_fern_kernel
    const int warmup_iterations = 50; // Number of initial iterations to discard
    std::cout << "Launching Barnsley Fern generation kernel..." << std::endl;
    // Kernel declaration is in barnsley_fern.cuh
    generate_fern_kernel<<<num_blocks, threads_per_block>>>(
        d_image_buffer, 
        image_width, 
        image_height, 
        num_points_per_thread, // This is actually iterations per thread that contribute to the image
        d_rand_states,
        x_min_fern, x_max_fern, y_min_fern, y_max_fern,
        warmup_iterations);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    std::cout << "Kernel execution finished." << std::endl;

    // Copy result back to host
    std::cout << "Copying image buffer from device to host..." << std::endl;
    CHECK_CUDA_ERROR(cudaMemcpy(h_image_buffer.data(), d_image_buffer, image_width * image_height * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    std::cout << "Copy finished." << std::endl;

    // Save image to PGM
    save_pgm("barnsley_fern.pgm", h_image_buffer.data(), image_width, image_height);
    std::cout << "Saved fern to barnsley_fern.pgm" << std::endl;

    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_image_buffer));
    CHECK_CUDA_ERROR(cudaFree(d_rand_states));

    return 0;
}

void save_pgm(const char* filename, const unsigned int* buffer, int width, int height) {
    std::ofstream outfile(filename, std::ios_base::out | std::ios_base::binary);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open PGM file for writing: " << filename << std::endl;
        return;
    }

    outfile << "P5\n";
    outfile << width << " " << height << "\n";
    outfile << 255 << "\n";

    unsigned int min_hit_val = 0; 
    unsigned int max_hit_val = 0; 
    bool any_pixel_hit = false;

    // First pass: find true min and max hit counts among *hit* pixels
    for (int i = 0; i < width * height; ++i) {
        if (buffer[i] > 0) {
            if (!any_pixel_hit) { 
                min_hit_val = buffer[i];
                max_hit_val = buffer[i];
                any_pixel_hit = true;
            } else { 
                if (buffer[i] < min_hit_val) min_hit_val = buffer[i];
                if (buffer[i] > max_hit_val) max_hit_val = buffer[i];
            }
        }
    }

    std::vector<unsigned char> char_buffer(width * height);
    for (int i = 0; i < width * height; ++i) {
        if (buffer[i] > 0) {
            char_buffer[i] = 255; // White if hit
        } else {
            char_buffer[i] = 0;   // Black if not hit
        }
    }

    outfile.write(reinterpret_cast<const char*>(char_buffer.data()), width * height);
    outfile.close();
}
