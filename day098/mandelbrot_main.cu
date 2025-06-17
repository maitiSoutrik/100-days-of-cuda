#include "mandelbrot_amr.cuh"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono> // For timing

// Helper function to save image as PGM
void save_pgm(const std::string& filename, const unsigned char* buffer, int width, int height) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error: Cannot open file for writing: " << filename << std::endl;
        return;
    }
    outfile << "P5\n";
    outfile << width << " " << height << "\n";
    outfile << "255\n";
    outfile.write(reinterpret_cast<const char*>(buffer), width * height * sizeof(unsigned char));
    outfile.close();
    std::cout << "Image saved as " << filename << std::endl;
}

int main(int argc, char** argv) {
    // Image parameters
    const int width = 800;
    const int height = 600;

    // Mandelbrot parameters
    const double x_min = -2.0;
    const double x_max = 1.0;
    const double y_min = -1.0;
    const double y_max = 1.0;
    const int max_iterations = 500;
    const int max_depth = 3; // Max recursion depth for AMR. Start with a small value.
                             // 0 means no refinement, 1 means one level of subdivision, etc.
    const float refinement_threshold = 0.1f; // Example: if normalized variance > 0.1, refine.

    std::cout << "Generating Mandelbrot set with AMR..." << std::endl;
    std::cout << "Image size: " << width << "x" << height << std::endl;
    std::cout << "Max iterations: " << max_iterations << std::endl;
    std::cout << "Max AMR depth: " << max_depth << std::endl;
    std::cout << "Refinement threshold: " << refinement_threshold << std::endl;


    std::vector<unsigned char> image_data_host(width * height);

    // Perform Mandelbrot generation
    auto start_time = std::chrono::high_resolution_clock::now();

    generate_mandelbrot_amr(image_data_host.data(), width, height,
                              x_min, x_max, y_min, y_max,
                              max_iterations, max_depth, refinement_threshold);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end_time - start_time;
    std::cout << "Mandelbrot generation took: " << duration_ms.count() << " ms" << std::endl;

    // Save the image
    std::string output_filename = "mandelbrot_amr_output.pgm";
    save_pgm(output_filename, image_data_host.data(), width, height);
    
    // Check for any CUDA errors that might have occurred during the run
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after kernel execution: " << cudaGetErrorString(err) << std::endl;
        // It's good practice to also call cudaDeviceReset() here if not done in CHECK_CUDA_ERROR
        // cudaDeviceReset(); // Uncomment if your CHECK_CUDA_ERROR doesn't always exit
        return 1;
    }
    
    std::cout << "Successfully generated Mandelbrot set." << std::endl;

    return 0;
}
