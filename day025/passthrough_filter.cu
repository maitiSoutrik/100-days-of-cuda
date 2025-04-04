#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <limits>
#include <chrono>
#include <cmath> // For std::fabs
#include <cuda_runtime.h> // Add CUDA runtime header

// CUDA error checking macro
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

// Simple Point structure
struct Point {
    float x, y, z;
};

// --- Host Functions for PCD I/O (Simplified ASCII XYZ) ---

// Reads XYZ data from a simple ASCII PCD file
bool readPCD(const std::string& filename, std::vector<Point>& points) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    std::string line;
    int point_count = 0;
    bool data_section = false;

    while (getline(file, line)) {
        if (line.empty() || line[0] == '#') continue; // Skip comments and empty lines

        std::stringstream ss(line);
        std::string keyword;
        ss >> keyword;

        if (keyword == "POINTS") {
            ss >> point_count;
            points.reserve(point_count);
        } else if (keyword == "DATA") {
            std::string type;
            ss >> type;
            if (type == "ascii") {
                data_section = true;
            } else {
                std::cerr << "Error: Only ASCII PCD data is supported." << std::endl;
                return false;
            }
        } else if (data_section) {
            // Assume lines after DATA ascii are point data
            std::stringstream data_ss(line);
            Point p;
            if (!(data_ss >> p.x >> p.y >> p.z)) {
                 // Allow for potential extra fields, just read XYZ
                 // std::cerr << "Warning: Could not parse point data: " << line << std::endl;
                 // continue; // Or handle more robustly
            }
             if (points.size() < point_count) { // Avoid reading past declared POINTS
                points.push_back(p);
            }
        }
        // Ignore other header lines like VERSION, FIELDS, SIZE, TYPE, COUNT, WIDTH, HEIGHT, VIEWPOINT
    }

    if (points.size() != point_count) {
         std::cerr << "Warning: Number of points read (" << points.size()
                   << ") does not match header (" << point_count << ")." << std::endl;
         // Adjust point_count if needed, or handle as error depending on strictness
         point_count = points.size();
    }


    std::cout << "Read " << points.size() << " points from " << filename << std::endl;
    return true;
}

// Writes XYZ data to a simple ASCII PCD file
bool writePCD(const std::string& filename, const std::vector<Point>& points) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return false;
    }

    file << "# .PCD v0.7 - Point Cloud Data file format\n";
    file << "VERSION .7\n";
    file << "FIELDS x y z\n";
    file << "SIZE 4 4 4\n";
    file << "TYPE F F F\n";
    file << "COUNT 1 1 1\n";
    file << "WIDTH " << points.size() << "\n";
    file << "HEIGHT 1\n"; // Unorganized point cloud
    file << "VIEWPOINT 0 0 0 1 0 0 0\n";
    file << "POINTS " << points.size() << "\n";
    file << "DATA ascii\n";

    for (const auto& p : points) {
        file << p.x << " " << p.y << " " << p.z << "\n";
    }

    std::cout << "Wrote " << points.size() << " points to " << filename << std::endl;
    return true;
}


// --- CUDA Kernels ---

__global__ void passthrough_filter_kernel(const Point* d_in_points, int* d_flags, int num_points,
                                          float min_x, float max_x,
                                          float min_y, float max_y,
                                          float min_z, float max_z)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_points) {
        Point p = d_in_points[idx];
        bool pass = (p.x >= min_x && p.x <= max_x &&
                     p.y >= min_y && p.y <= max_y &&
                     p.z >= min_z && p.z <= max_z);
        d_flags[idx] = pass ? 1 : 0;
    }
}

// Simple compaction kernel using atomicAdd to determine output indices
__global__ void compact_points_kernel(const Point* d_in_points, const int* d_flags,
                                      Point* d_out_points, int* d_out_count, int num_points)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Use shared memory for intermediate count for potential small optimization
    // extern __shared__ int smem_count[]; // Requires dynamic shared memory allocation
    // if (threadIdx.x == 0) smem_count[0] = 0;
    // __syncthreads();

    if (idx < num_points && d_flags[idx] == 1) {
        // Atomically increment the output counter and get the index for this point
        int out_idx = atomicAdd(d_out_count, 1);
        // int out_idx = atomicAdd(&smem_count[0], 1); // If using shared memory per block

        // Check bounds just in case (though atomicAdd should manage this)
        // if (out_idx < *d_out_count) { // This check is tricky with atomics
            d_out_points[out_idx] = d_in_points[idx];
        // }
    }
    // __syncthreads();
    // if (threadIdx.x == 0) {
    //     atomicAdd(d_global_out_count, smem_count[0]); // Add block count to global if using shared
    // }
}


// --- Main Function ---

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0] << " <input.pcd> <output.pcd> <min_x> <max_x> <min_y> <max_y>" << std::endl;
        // Simplified usage: only filter on X for this example
        std::cerr << "Simplified Usage: " << argv[0] << " <input.pcd> <output.pcd> <min_x> <max_x> -inf inf" << std::endl;
        return 1;
    }

    std::string input_filename = argv[1];
    std::string output_filename = argv[2];
    float min_x = -std::numeric_limits<float>::infinity();
    float max_x = std::numeric_limits<float>::infinity();
    float min_y = -std::numeric_limits<float>::infinity();
    float max_y = std::numeric_limits<float>::infinity();
    float min_z = -std::numeric_limits<float>::infinity();
    float max_z = std::numeric_limits<float>::infinity();

    try {
        min_x = std::stof(argv[3]);
        max_x = std::stof(argv[4]);
        // Allow "inf" or "-inf" for bounds
        if (std::string(argv[5]) != "-inf") min_y = std::stof(argv[5]);
        if (std::string(argv[6]) != "inf") max_y = std::stof(argv[6]);
        // For this example, we'll keep Z bounds infinite, but could add args later
        // min_z = std::stof(argv[7]);
        // max_z = std::stof(argv[8]);
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error parsing filter bounds: " << e.what() << std::endl;
        return 1;
    } catch (const std::out_of_range& e) {
         std::cerr << "Error parsing filter bounds (out of range): " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Input file: " << input_filename << std::endl;
    std::cout << "Output file: " << output_filename << std::endl;
    std::cout << "Filter bounds: X[" << min_x << ", " << max_x << "], Y["
              << min_y << ", " << max_y << "], Z[" << min_z << ", " << max_z << "]" << std::endl;


    // 1. Read PCD file (Host)
    std::vector<Point> h_in_points;
    if (!readPCD(input_filename, h_in_points)) {
        return 1;
    }
    if (h_in_points.empty()) {
        std::cerr << "Error: No points read from input file." << std::endl;
        return 1;
    }
    int num_points = h_in_points.size();

    // 2. Allocate CUDA Memory
    Point* d_in_points = nullptr;
    Point* d_out_points = nullptr;
    int* d_flags = nullptr;
    int* d_out_count = nullptr; // Counter for filtered points on device
    int h_out_count = 0;       // Counter for filtered points on host

    CHECK_CUDA_ERROR(cudaMalloc(&d_in_points, num_points * sizeof(Point)));
    // Allocate output buffer potentially as large as input
    CHECK_CUDA_ERROR(cudaMalloc(&d_out_points, num_points * sizeof(Point)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_flags, num_points * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_out_count, sizeof(int)));

    // 3. Copy Input Data to Device
    CHECK_CUDA_ERROR(cudaMemcpy(d_in_points, h_in_points.data(), num_points * sizeof(Point), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemset(d_flags, 0, num_points * sizeof(int))); // Initialize flags to 0
    CHECK_CUDA_ERROR(cudaMemset(d_out_count, 0, sizeof(int)));         // Initialize output count to 0

    // 4. Define Kernel Launch Parameters
    int threads_per_block = 256;
    int blocks_per_grid = (num_points + threads_per_block - 1) / threads_per_block;

    std::cout << "Launching kernels with " << blocks_per_grid << " blocks and "
              << threads_per_block << " threads per block." << std::endl;

    // 5. Launch Kernels
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaEventRecord(start));

    // Kernel 1: Mark points that pass the filter
    passthrough_filter_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_in_points, d_flags, num_points, min_x, max_x, min_y, max_y, min_z, max_z);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors

    // Kernel 2: Compact points into the output buffer
    compact_points_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_in_points, d_flags, d_out_points, d_out_count, num_points);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors

    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "GPU filtering and compaction took: " << milliseconds << " ms" << std::endl;

    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));


    // 6. Copy Results Back to Host
    CHECK_CUDA_ERROR(cudaMemcpy(&h_out_count, d_out_count, sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<Point> h_out_points(h_out_count); // Resize host vector based on actual count
    if (h_out_count > 0) {
        CHECK_CUDA_ERROR(cudaMemcpy(h_out_points.data(), d_out_points, h_out_count * sizeof(Point), cudaMemcpyDeviceToHost));
    }

    std::cout << "Number of points after filtering: " << h_out_count << std::endl;

    // 7. Write Output PCD File (Host)
    if (!writePCD(output_filename, h_out_points)) {
        // Error already printed in writePCD
    }

    // 8. Free CUDA Memory
    CHECK_CUDA_ERROR(cudaFree(d_in_points));
    CHECK_CUDA_ERROR(cudaFree(d_out_points));
    CHECK_CUDA_ERROR(cudaFree(d_flags));
    CHECK_CUDA_ERROR(cudaFree(d_out_count));

    std::cout << "Passthrough filter completed successfully." << std::endl;

    return 0;
}
