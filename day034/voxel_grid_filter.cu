#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <limits> // Required for numeric_limits
#include <cmath> // For ceilf, floorf
#include <stdexcept> // For runtime_error
#include <chrono>

// CUDA runtime
#include <cuda_runtime.h>

// Helper macro for checking CUDA errors
#define CHECK_CUDA_ERROR(val) checkCudaError((val), #val, __FILE__, __LINE__)
inline void checkCudaError(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(err) << "(" << cudaGetErrorString(err) << ") \"" << func << "\"" << std::endl;
        cudaDeviceReset(); // Reset device on error
        exit(EXIT_FAILURE);
    }
}

// Structure to hold point data
struct Point {
    float x, y, z;
};

// Structure for voxel data (used internally in the kernel potentially, or on host)
// We might use separate arrays for sums and counts directly for atomics
struct VoxelData {
    float sum_x, sum_y, sum_z;
    int count;
};


// Forward declarations for CUDA kernels
__global__ void voxel_grid_hash_kernel(/* params */);
__global__ void compute_centroids_kernel(/* params */);


// Host function to parse ASCII PCD file header and data
bool loadPCDFile(const std::string& filename, std::vector<Point>& points, Point& min_pt, Point& max_pt) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open PCD file: " << filename << std::endl;
        return false;
    }

    std::string line;
    int num_points = 0;
    bool data_ascii = false;
    bool header_parsed = false;

    min_pt = { std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
    max_pt = { std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest() };

    // Parse header
    while (std::getline(file, line) && !header_parsed) {
        std::stringstream ss(line);
        std::string type;
        ss >> type;

        if (type == "POINTS") {
            ss >> num_points;
        } else if (type == "DATA") {
            std::string format;
            ss >> format;
            if (format == "ascii") {
                data_ascii = true;
            } else {
                std::cerr << "Error: Only ASCII PCD format is supported." << std::endl;
                return false;
            }
            header_parsed = true; // End of header
        }
        // Ignore other header lines (VERSION, FIELDS, SIZE, TYPE, COUNT, WIDTH, HEIGHT, VIEWPOINT)
    }

    if (!data_ascii || num_points <= 0) {
        std::cerr << "Error: Invalid PCD header or non-ASCII data." << std::endl;
        return false;
    }

    points.reserve(num_points);

    // Parse data
    while (std::getline(file, line) && points.size() < num_points) {
        std::stringstream ss(line);
        Point p;
        // Assuming format is X Y Z ... (ignore potential intensity, etc.)
        if (ss >> p.x >> p.y >> p.z) {
            points.push_back(p);
            // Update min/max bounds
            min_pt.x = fminf(min_pt.x, p.x);
            min_pt.y = fminf(min_pt.y, p.y);
            min_pt.z = fminf(min_pt.z, p.z);
            max_pt.x = fmaxf(max_pt.x, p.x);
            max_pt.y = fmaxf(max_pt.y, p.y);
            max_pt.z = fmaxf(max_pt.z, p.z);
        } else {
             // Tolerate empty lines sometimes found at the end of files
            if (!line.empty()) {
                 std::cerr << "Warning: Could not parse point data line: " << line << std::endl;
            }
        }
    }

     if (points.size() != num_points) {
        std::cerr << "Warning: Number of points read (" << points.size() << ") does not match header (" << num_points << ")." << std::endl;
        // Continue with the points read if any were successful
         if (points.empty()) return false;
    }


    std::cout << "Successfully loaded " << points.size() << " points from " << filename << std::endl;
    std::cout << "Bounding Box: " << std::endl;
    std::cout << "  Min: (" << min_pt.x << ", " << min_pt.y << ", " << min_pt.z << ")" << std::endl;
    std::cout << "  Max: (" << max_pt.x << ", " << max_pt.y << ", " << max_pt.z << ")" << std::endl;

    return true;
}

// Host function to save filtered points to a simple text file (optional)
void saveFilteredPoints(const std::string& filename, const std::vector<Point>& points) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open output file: " << filename << std::endl;
        return;
    }
    file << "# X Y Z - Filtered Point Cloud" << std::endl;
    for (const auto& p : points) {
        file << p.x << " " << p.y << " " << p.z << std::endl;
    }
    std::cout << "Saved " << points.size() << " filtered points to " << filename << std::endl;
}

// --- KERNEL IMPLEMENTATIONS ---

// Kernel 1: Calculate voxel index for each point and accumulate sums/counts using atomics
__global__ void voxel_grid_hash_kernel(const Point* d_points, int num_points,
                                       float* d_voxel_sum_x, float* d_voxel_sum_y, float* d_voxel_sum_z,
                                       int* d_voxel_counts,
                                       Point min_bounds, Point max_bounds,
                                       float voxel_size_x, float voxel_size_y, float voxel_size_z,
                                       int grid_dim_x, int grid_dim_y, int grid_dim_z) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_points) {
        return;
    }

    Point p = d_points[idx];

    // Calculate voxel index (handle potential floating point inaccuracies slightly differently from strict floor)
    // Ensure indices are within bounds [0, grid_dim - 1]
    int voxel_idx_x = static_cast<int>(floorf((p.x - min_bounds.x) / voxel_size_x));
    int voxel_idx_y = static_cast<int>(floorf((p.y - min_bounds.y) / voxel_size_y));
    int voxel_idx_z = static_cast<int>(floorf((p.z - min_bounds.z) / voxel_size_z));

    // Clamp indices to be within the grid dimensions just in case (e.g., point exactly on max boundary)
    voxel_idx_x = max(0, min(voxel_idx_x, grid_dim_x - 1));
    voxel_idx_y = max(0, min(voxel_idx_y, grid_dim_y - 1));
    voxel_idx_z = max(0, min(voxel_idx_z, grid_dim_z - 1));

    // Calculate 1D linear index
    int linear_voxel_idx = voxel_idx_z * (grid_dim_x * grid_dim_y) + voxel_idx_y * grid_dim_x + voxel_idx_x;

    // Use atomicAdd to safely update the sums and count for this voxel
    atomicAdd(&d_voxel_sum_x[linear_voxel_idx], p.x);
    atomicAdd(&d_voxel_sum_y[linear_voxel_idx], p.y);
    atomicAdd(&d_voxel_sum_z[linear_voxel_idx], p.z);
    atomicAdd(&d_voxel_counts[linear_voxel_idx], 1);
}

// Kernel 2: Compute centroids from accumulated sums and counts
__global__ void compute_centroids_kernel(const float* d_voxel_sum_x, const float* d_voxel_sum_y, const float* d_voxel_sum_z,
                                        const int* d_voxel_counts,
                                        Point* d_filtered_points,
                                        int* d_num_filtered_points, // Use atomic to count actual output points
                                        int total_voxels) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_voxels) {
        return;
    }

    int count = d_voxel_counts[idx];
    if (count > 0) {
        // Calculate centroid
        Point centroid;
        centroid.x = d_voxel_sum_x[idx] / count;
        centroid.y = d_voxel_sum_y[idx] / count;
        centroid.z = d_voxel_sum_z[idx] / count;

        // Atomically get the index to write the output point and increment the counter
        int output_idx = atomicAdd(d_num_filtered_points, 1);

        // Write the centroid to the output array
        // Make sure d_filtered_points is large enough (allocated for total_voxels worst case)
        d_filtered_points[output_idx] = centroid;
    }
}


// --- MAIN FUNCTION ---
int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <pcd_file> <voxel_size_x> <voxel_size_y> <voxel_size_z>" << std::endl;
        return 1;
    }

    std::string pcd_filename = argv[1];
    float voxel_size_x = std::stof(argv[2]);
    float voxel_size_y = std::stof(argv[3]);
    float voxel_size_z = std::stof(argv[4]);

    if (voxel_size_x <= 0 || voxel_size_y <= 0 || voxel_size_z <= 0) {
        std::cerr << "Error: Voxel sizes must be positive." << std::endl;
        return 1;
    }
    Point voxel_size = {voxel_size_x, voxel_size_y, voxel_size_z};
    std::cout << "Using Voxel Size: (" << voxel_size.x << ", " << voxel_size.y << ", " << voxel_size.z << ")" << std::endl;

    // --- 1. Load Data (Host) ---
    std::vector<Point> h_points;
    Point min_bounds, max_bounds;
    auto start_load = std::chrono::high_resolution_clock::now();
    if (!loadPCDFile(pcd_filename, h_points, min_bounds, max_bounds)) {
        return 1;
    }
    auto end_load = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> load_duration = end_load - start_load;
    std::cout << "PCD Loading time: " << load_duration.count() << " ms" << std::endl;


    if (h_points.empty()) {
         std::cerr << "Error: No points loaded from PCD file." << std::endl;
         return 1;
    }
    int num_points = h_points.size();

    // --- 2. Calculate Grid Dimensions (Host) ---
    // Add a small epsilon to max_bounds to handle points exactly on the boundary
    float epsilon = 1e-4f;
    int grid_dim_x = static_cast<int>(ceilf((max_bounds.x - min_bounds.x) / voxel_size.x));
    int grid_dim_y = static_cast<int>(ceilf((max_bounds.y - min_bounds.y) / voxel_size.y));
    int grid_dim_z = static_cast<int>(ceilf((max_bounds.z - min_bounds.z) / voxel_size.z));

    // Ensure grid dimensions are at least 1
    grid_dim_x = std::max(1, grid_dim_x);
    grid_dim_y = std::max(1, grid_dim_y);
    grid_dim_z = std::max(1, grid_dim_z);

    size_t total_voxels = static_cast<size_t>(grid_dim_x) * grid_dim_y * grid_dim_z;

    std::cout << "Grid Dimensions: (" << grid_dim_x << ", " << grid_dim_y << ", " << grid_dim_z << ")" << std::endl;
    std::cout << "Total Voxels: " << total_voxels << std::endl;

    if (total_voxels == 0) {
        std::cerr << "Error: Calculated zero total voxels. Check input data or voxel size." << std::endl;
        return 1;
    }
     // Add a check for potentially huge memory allocation
    const size_t max_reasonable_voxels = 100 * 1024 * 1024; // ~100 million voxels, adjust as needed
    if (total_voxels > max_reasonable_voxels) {
        std::cerr << "Warning: Very large number of voxels (" << total_voxels << "). This may require significant memory (> "
                  << (total_voxels * (sizeof(float) * 3 + sizeof(int)) >> 20) << " MB)." << std::endl;
        // Consider adding a confirmation step or exiting if it's too large
        // For now, just warn.
    }


    // --- 3. Allocate Memory (Device) ---
    Point* d_points = nullptr;
    float* d_voxel_sum_x = nullptr;
    float* d_voxel_sum_y = nullptr;
    float* d_voxel_sum_z = nullptr;
    int*   d_voxel_counts = nullptr;
    Point* d_filtered_points = nullptr; // Output buffer (worst case size = total_voxels)
    int*   d_num_filtered_points = nullptr; // Counter for actual output points

    CHECK_CUDA_ERROR(cudaMalloc(&d_points, num_points * sizeof(Point)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_voxel_sum_x, total_voxels * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_voxel_sum_y, total_voxels * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_voxel_sum_z, total_voxels * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_voxel_counts, total_voxels * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_filtered_points, total_voxels * sizeof(Point))); // Allocate worst-case size
    CHECK_CUDA_ERROR(cudaMalloc(&d_num_filtered_points, sizeof(int)));

    // --- 4. Initialize Device Memory ---
    CHECK_CUDA_ERROR(cudaMemset(d_voxel_sum_x, 0, total_voxels * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_voxel_sum_y, 0, total_voxels * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_voxel_sum_z, 0, total_voxels * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_voxel_counts, 0, total_voxels * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(d_num_filtered_points, 0, sizeof(int)));

    // --- 5. Transfer Data (Host to Device) ---
    auto start_h2d = std::chrono::high_resolution_clock::now();
    CHECK_CUDA_ERROR(cudaMemcpy(d_points, h_points.data(), num_points * sizeof(Point), cudaMemcpyHostToDevice));
    auto end_h2d = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> h2d_duration = end_h2d - start_h2d;


    // --- 6. Launch Kernels ---
    int threads_per_block = 256;

    // Kernel 1: Hashing
    int blocks_hash = (num_points + threads_per_block - 1) / threads_per_block;
    auto start_kernel1 = std::chrono::high_resolution_clock::now();
    voxel_grid_hash_kernel<<<blocks_hash, threads_per_block>>>(
        d_points, num_points,
        d_voxel_sum_x, d_voxel_sum_y, d_voxel_sum_z, d_voxel_counts,
        min_bounds, max_bounds,
        voxel_size.x, voxel_size.y, voxel_size.z,
        grid_dim_x, grid_dim_y, grid_dim_z
    );
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for kernel 1 to complete
    auto end_kernel1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> kernel1_duration = end_kernel1 - start_kernel1;


    // Kernel 2: Centroid Calculation
    int blocks_centroid = (total_voxels + threads_per_block - 1) / threads_per_block;
    auto start_kernel2 = std::chrono::high_resolution_clock::now();
    compute_centroids_kernel<<<blocks_centroid, threads_per_block>>>(
        d_voxel_sum_x, d_voxel_sum_y, d_voxel_sum_z, d_voxel_counts,
        d_filtered_points, d_num_filtered_points,
        total_voxels
    );
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for kernel 2 to complete
    auto end_kernel2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> kernel2_duration = end_kernel2 - start_kernel2;

    // --- 7. Transfer Results (Device to Host) ---
    int h_num_filtered_points = 0;
    auto start_d2h = std::chrono::high_resolution_clock::now();
    CHECK_CUDA_ERROR(cudaMemcpy(&h_num_filtered_points, d_num_filtered_points, sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<Point> h_filtered_points;
    if (h_num_filtered_points > 0) {
        h_filtered_points.resize(h_num_filtered_points);
        CHECK_CUDA_ERROR(cudaMemcpy(h_filtered_points.data(), d_filtered_points, h_num_filtered_points * sizeof(Point), cudaMemcpyDeviceToHost));
    }
    auto end_d2h = std::chrono::high_resolution_clock::now();
     std::chrono::duration<double, std::milli> d2h_duration = end_d2h - start_d2h;


    // --- 8. Output Results and Timings ---
    std::cout << "\n--- Voxel Grid Filter Results ---" << std::endl;
    std::cout << "Input points: " << num_points << std::endl;
    std::cout << "Output points (centroids): " << h_num_filtered_points << std::endl;

     std::cout << "\n--- Performance Timings ---" << std::endl;
    std::cout << "Host->Device Transfer: " << h2d_duration.count() << " ms" << std::endl;
    std::cout << "Voxel Hash Kernel:     " << kernel1_duration.count() << " ms" << std::endl;
    std::cout << "Centroid Kernel:       " << kernel2_duration.count() << " ms" << std::endl;
    std::cout << "Device->Host Transfer: " << d2h_duration.count() << " ms" << std::endl;
    std::cout << "Total GPU processing (Kernels + D2H): " << kernel1_duration.count() + kernel2_duration.count() + d2h_duration.count() << " ms" << std::endl;


    // --- 9. Save Filtered Points (Optional) ---
    // saveFilteredPoints("filtered_points.txt", h_filtered_points);


    // --- 10. Cleanup ---
    CHECK_CUDA_ERROR(cudaFree(d_points));
    CHECK_CUDA_ERROR(cudaFree(d_voxel_sum_x));
    CHECK_CUDA_ERROR(cudaFree(d_voxel_sum_y));
    CHECK_CUDA_ERROR(cudaFree(d_voxel_sum_z));
    CHECK_CUDA_ERROR(cudaFree(d_voxel_counts));
    CHECK_CUDA_ERROR(cudaFree(d_filtered_points));
    CHECK_CUDA_ERROR(cudaFree(d_num_filtered_points));

    // Optional: Reset device if desired at the end of the application
    // CHECK_CUDA_ERROR(cudaDeviceReset());

    std::cout << "\nVoxel grid filtering completed successfully." << std::endl;

    return 0;
}
