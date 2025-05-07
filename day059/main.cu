#include "ray_tracer.cuh" // Includes cuda_runtime.h, cmath, Vec3, kernel declaration, constants
#include <stdio.h>
#include <stdlib.h>
#include <string> // For std::string
#include <sys/stat.h> // For mkdir

// Function to ensure the output directory exists
void ensure_output_directory(const char* dir_path) {
#if defined(_WIN32)
    _mkdir(dir_path); // For Windows
#else
    mkdir(dir_path, 0755); // For POSIX systems (Linux, macOS)
#endif
    // No error checking for simplicity, assuming it usually works or path is valid.
    // In a robust app, you'd check mkdir's return value.
}


// Save image as PPM
void save_image(unsigned char *image, const char* filename) {
    // Ensure the "output" directory exists before trying to write the file
    // Assuming filename is "output/output.ppm" or similar.
    // Let's extract directory from filename.
    std::string filepath_str(filename);
    size_t last_slash_idx = filepath_str.find_last_of("/\\");
    if (std::string::npos != last_slash_idx) {
        std::string directory = filepath_str.substr(0, last_slash_idx);
        if (!directory.empty()) {
            ensure_output_directory(directory.c_str());
        }
    }

    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Error: Could not open file %s for writing.\n", filename);
        return;
    }
    fprintf(f, "P6\n%d %d\n255\n", WIDTH, HEIGHT);
    fwrite(image, 1, WIDTH * HEIGHT * 3, f);
    fclose(f);
    printf("Image saved as %s\n", filename);
}

int main() {
    // Allocate memory for image
    unsigned char *d_image, *h_image;
    size_t image_size = WIDTH * HEIGHT * 3 * sizeof(unsigned char); // Corrected size

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_image, image_size));
    
    h_image = (unsigned char *)malloc(image_size);
    if (h_image == NULL) {
        fprintf(stderr, "Error: Failed to allocate host memory for image.\n");
        cudaFree(d_image);
        return EXIT_FAILURE;
    }

    // Launch CUDA kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    printf("Launching render kernel with %u blocks and %u threads per block...\n", numBlocks.x * numBlocks.y, threadsPerBlock.x * threadsPerBlock.y);
    render<<<numBlocks, threadsPerBlock>>>(d_image);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for kernel to complete

    // Copy image from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost));

    // Save image to output/output.ppm as per .clinerules
    // The CMakeLists.txt creates build/day059/output if it doesn't exist when building.
    // This save_image will attempt to create ./output/ relative to executable CWD.
    // When run by ctest or CI, CWD is usually build/dayXXX/.
    // So this will create build/dayXXX/output/output.ppm
    save_image(h_image, "output/output.ppm");

    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_image));
    free(h_image);

    printf("Day 059: Ray Tracing with CUDA completed.\n");
    return EXIT_SUCCESS;
}
