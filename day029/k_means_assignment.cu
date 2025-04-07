#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <limits> // Required for std::numeric_limits
#include <cmath> // For std::sqrt, std::fabs
#include <cfloat> // For FLT_MAX
#include <getopt.h> // For argument parsing
#include <cstdlib> // For exit, EXIT_FAILURE
#include <chrono> // For timing
#include <iomanip> // For std::setprecision

// CUDA includes
#include <cuda_runtime.h>
#include <vector_types.h> // For float2

// --- Helper Macros ---
#define CHECK_CUDA_ERROR(val) checkCudaError((val), #val, __FILE__, __LINE__)
inline void checkCudaError(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error at " << file << ":" << line << " code=" << static_cast<unsigned int>(err) << " \"" << cudaGetErrorString(err) << "\" for " << func << std::endl;
        exit(EXIT_FAILURE);
    }
}

// --- Forward Declarations ---
void print_usage(const char* prog_name);
bool load_points_from_file(const std::string& filename, std::vector<float2>& points);
void generate_synthetic_data(std::vector<float2>& points, int num_points, int num_clusters);
void initialize_centroids(std::vector<float2>& centroids, int num_clusters, const std::vector<float2>& points);
void assign_points_cpu(const std::vector<float2>& points, const std::vector<float2>& centroids, std::vector<int>& assignments_cpu);
bool save_assignments(const std::string& filename, const int* assignments, int num_points);

// --- CUDA Kernel ---
__global__ void assign_points_kernel(
    const float2* d_points,
    const float2* d_centroids,
    int* d_assignments,
    int num_points,
    int num_clusters
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    float2 my_point = d_points[idx];
    float min_dist_sq = FLT_MAX;
    int best_cluster_idx = -1;

    // Loop through centroids to find the closest one
    for (int k = 0; k < num_clusters; k++) {
        float2 centroid_k = d_centroids[k];
        float dx = my_point.x - centroid_k.x;
        float dy = my_point.y - centroid_k.y;
        float dist_sq = dx * dx + dy * dy; // Squared Euclidean distance

        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            best_cluster_idx = k;
        }
    }
    d_assignments[idx] = best_cluster_idx;
}

// --- Main Function ---
int main(int argc, char* argv[]) {
    // Default parameters
    int num_clusters = 0;
    std::string mode = "synthetic";
    int num_points_synthetic = 1000; // Default if mode is synthetic
    std::string input_filepath = "";
    std::string output_filepath = "";
    bool display_help = false;

    // --- Argument Parsing ---
    const struct option long_options[] = {
        {"clusters", required_argument, nullptr, 'k'},
        {"mode", required_argument, nullptr, 'm'}, // Changed to required_argument
        {"points", required_argument, nullptr, 'p'},
        {"input", required_argument, nullptr, 'i'},
        {"output", optional_argument, nullptr, 'o'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0} // End of options
    };

    int opt;
    int long_index = 0;
    // Changed "m::" to "m:" in the short options string
    while ((opt = getopt_long(argc, argv, "k:m:p:i:o::h", long_options, &long_index)) != -1) {
        switch (opt) {
            case 'k': num_clusters = std::stoi(optarg); break;
            case 'm': mode = optarg; break; // Argument is now required
            case 'p': num_points_synthetic = std::stoi(optarg); break;
            case 'i': input_filepath = optarg; break;
            case 'o': output_filepath = (optarg ? optarg : "assignments_output.txt"); break; // Handle optional argument
            case 'h': display_help = true; break;
            default: print_usage(argv[0]); return EXIT_FAILURE;
        }
    }

    if (display_help) {
        print_usage(argv[0]);
        return EXIT_SUCCESS;
    }

    // Validate arguments
    if (num_clusters <= 0) {
        std::cerr << "Error: Number of clusters (--clusters) must be a positive integer." << std::endl;
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    if (mode != "synthetic" && mode != "file") {
        std::cerr << "Error: Invalid mode '" << mode << "'. Must be 'synthetic' or 'file'." << std::endl;
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    if (mode == "file" && input_filepath.empty()) {
        std::cerr << "Error: Input file path (--input) is required for 'file' mode." << std::endl;
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
     if (mode == "synthetic" && num_points_synthetic <= 0) {
        std::cerr << "Error: Number of points (--points) must be positive for 'synthetic' mode." << std::endl;
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    std::cout << "Parsed Arguments:" << std::endl;
    std::cout << "  Mode: " << mode << std::endl;
    std::cout << "  Clusters: " << num_clusters << std::endl;
    if (mode == "synthetic") {
        std::cout << "  Synthetic Points: " << num_points_synthetic << std::endl;
    } else {
        std::cout << "  Input File: " << input_filepath << std::endl;
    }
    if (!output_filepath.empty()) {
        std::cout << "  Output File: " << output_filepath << std::endl;
    }

    // --- Host Data Structures ---
    std::vector<float2> h_points;
    std::vector<float2> h_centroids(num_clusters);
    int num_points = 0;

    // --- Load or Generate Data ---
    if (mode == "file") {
        std::cout << "Loading points from file: " << input_filepath << std::endl;
        if (!load_points_from_file(input_filepath, h_points)) {
            std::cerr << "Error loading points from file." << std::endl;
            return EXIT_FAILURE;
        }
        num_points = h_points.size();
        if (num_points == 0) {
             std::cerr << "Error: No points loaded from file " << input_filepath << std::endl;
             return EXIT_FAILURE;
        }
        std::cout << "Loaded " << num_points << " points." << std::endl;
    } else { // mode == "synthetic"
        num_points = num_points_synthetic;
        h_points.resize(num_points);
        std::cout << "Generating " << num_points << " synthetic points for " << num_clusters << " clusters..." << std::endl;
        generate_synthetic_data(h_points, num_points, num_clusters);
        std::cout << "Generated " << num_points << " points." << std::endl;
    }

    // --- Initialize Centroids ---
    std::cout << "Initializing " << num_clusters << " centroids randomly within data bounds..." << std::endl;
    initialize_centroids(h_centroids, num_clusters, h_points);

    // --- Host Memory for Results ---
    std::vector<int> h_assignments_gpu(num_points);
    std::vector<int> h_assignments_cpu(num_points);

    // --- Device Memory Allocation ---
    std::cout << "Allocating memory..." << std::endl;
    float2* d_points = nullptr;
    float2* d_centroids = nullptr;
    int* d_assignments = nullptr;
    size_t points_size = num_points * sizeof(float2);
    size_t centroids_size = num_clusters * sizeof(float2);
    size_t assignments_size = num_points * sizeof(int);

    CHECK_CUDA_ERROR(cudaMalloc(&d_points, points_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_centroids, centroids_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_assignments, assignments_size));

    // --- Data Transfer: Host -> Device ---
    std::cout << "Copying data to device..." << std::endl;
    CHECK_CUDA_ERROR(cudaMemcpy(d_points, h_points.data(), points_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_centroids, h_centroids.data(), centroids_size, cudaMemcpyHostToDevice));

    // --- Kernel Launch ---
    std::cout << "Launching CUDA kernel..." << std::endl;
    int block_size = 256;
    int grid_size = (num_points + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    assign_points_kernel<<<grid_size, block_size>>>(d_points, d_centroids, d_assignments, num_points, num_clusters);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop)); // Wait for kernel completion

    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "GPU Kernel Execution Time: " << milliseconds << " ms" << std::endl;

    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));


    // --- Data Transfer: Device -> Host ---
    std::cout << "Copying results from device..." << std::endl;
    CHECK_CUDA_ERROR(cudaMemcpy(h_assignments_gpu.data(), d_assignments, assignments_size, cudaMemcpyDeviceToHost));

    // --- Verification ---
    std::cout << "Running CPU verification..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    assign_points_cpu(h_points, h_centroids, h_assignments_cpu);
    auto stop_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(stop_cpu - start_cpu);
    std::cout << "CPU Verification Time: " << duration_cpu.count() << " ms" << std::endl;

    std::cout << "Comparing GPU and CPU results..." << std::endl;
    bool success = true;
    int mismatch_count = 0;
    for (int i = 0; i < num_points; ++i) {
        if (h_assignments_gpu[i] != h_assignments_cpu[i]) {
            if (mismatch_count < 10) { // Print first few mismatches
                 std::cerr << "Mismatch at index " << i << ": GPU=" << h_assignments_gpu[i]
                           << ", CPU=" << h_assignments_cpu[i] << std::endl;
            }
            success = false;
            mismatch_count++;
        }
    }

    if (success) {
        std::cout << "Verification Successful! All " << num_points << " assignments match." << std::endl;
    } else {
        std::cout << "Verification FAILED! " << mismatch_count << " assignments mismatch." << std::endl;
    }

    // --- Output Results (Optional) ---
    if (!output_filepath.empty()) {
        std::cout << "Saving assignments to " << output_filepath << "..." << std::endl;
        if (!save_assignments(output_filepath, h_assignments_gpu.data(), num_points)) {
             std::cerr << "Error saving assignments to file." << std::endl;
             // Continue cleanup even if saving fails
        }
    }

    // --- Cleanup ---
    std::cout << "Cleaning up memory..." << std::endl;
    CHECK_CUDA_ERROR(cudaFree(d_assignments));
    CHECK_CUDA_ERROR(cudaFree(d_centroids));
    CHECK_CUDA_ERROR(cudaFree(d_points));

    // Host vectors clean up automatically

    std::cout << "Done." << std::endl;
    return EXIT_SUCCESS;
}


// --- Helper Function Implementations ---

void print_usage(const char* prog_name) {
    std::cerr << "Usage: " << prog_name << " --clusters <K> [options]" << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -k, --clusters <K>    (Required) Number of clusters (centroids)." << std::endl;
    std::cerr << "  -m, --mode <mode>     (Optional) 'synthetic' or 'file'. Defaults to 'synthetic' if omitted." << std::endl; // Updated help text
    std::cerr << "  -p, --points <N>      (Required if mode=synthetic) Number of synthetic points." << std::endl;
    std::cerr << "  -i, --input <path>    (Required if mode=file) Path to input data file (X Y per line)." << std::endl;
    std::cerr << "  -o, --output [path]   (Optional) Path to save assignments (default: assignments_output.txt if flag present)." << std::endl;
    std::cerr << "  -h, --help            Display this usage information." << std::endl;
}

bool load_points_from_file(const std::string& filename, std::vector<float2>& points) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file: " << filename << std::endl;
        return false;
    }

    points.clear();
    std::string line;
    float x, y;
    int line_num = 0;
    while (std::getline(infile, line)) {
        line_num++;
        std::stringstream ss(line);
        if (!(ss >> x >> y)) {
            std::cerr << "Warning: Could not parse coordinates on line " << line_num << ". Skipping." << std::endl;
            continue; // Skip malformed lines
        }
        // Ignore rest of the line
        points.push_back(make_float2(x, y));
    }

    if (points.empty() && line_num > 0) {
         std::cerr << "Error: File opened but no valid points were parsed from " << filename << std::endl;
         return false;
    }
    if (points.empty() && line_num == 0) {
         std::cerr << "Warning: Input file " << filename << " was empty." << std::endl;
         // Allow empty files, main function checks for zero points loaded
    }

    return true;
}

void generate_synthetic_data(std::vector<float2>& points, int num_points, int num_clusters) {
    std::mt19937 gen(std::random_device{}()); // Mersenne Twister PRNG
    std::uniform_real_distribution<float> distrib_centers(-10.0f, 10.0f);
    std::normal_distribution<float> distrib_noise(0.0f, 1.5f); // Gaussian noise

    // Generate true centers
    std::vector<float2> true_centers(num_clusters);
    for (int k = 0; k < num_clusters; ++k) {
        true_centers[k] = make_float2(distrib_centers(gen), distrib_centers(gen));
    }

    // Generate points around true centers
    std::uniform_int_distribution<int> distrib_cluster_choice(0, num_clusters - 1);
    for (int i = 0; i < num_points; ++i) {
        int cluster_idx = distrib_cluster_choice(gen);
        float2 center = true_centers[cluster_idx];
        points[i] = make_float2(center.x + distrib_noise(gen), center.y + distrib_noise(gen));
    }
}

void initialize_centroids(std::vector<float2>& centroids, int num_clusters, const std::vector<float2>& points) {
    if (points.empty()) {
        std::cerr << "Error: Cannot initialize centroids from empty point set." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Find data bounds
    float min_x = points[0].x, max_x = points[0].x;
    float min_y = points[0].y, max_y = points[0].y;
    for (size_t i = 1; i < points.size(); ++i) {
        min_x = std::min(min_x, points[i].x);
        max_x = std::max(max_x, points[i].x);
        min_y = std::min(min_y, points[i].y);
        max_y = std::max(max_y, points[i].y);
    }

    // Initialize centroids randomly within bounds
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> distrib_x(min_x, max_x);
    std::uniform_real_distribution<float> distrib_y(min_y, max_y);

    centroids.resize(num_clusters);
    for (int k = 0; k < num_clusters; ++k) {
        centroids[k] = make_float2(distrib_x(gen), distrib_y(gen));
    }
}

void assign_points_cpu(const std::vector<float2>& points, const std::vector<float2>& centroids, std::vector<int>& assignments_cpu) {
    int num_points = points.size();
    int num_clusters = centroids.size();
    assignments_cpu.resize(num_points);

    #pragma omp parallel for // Optional: Use OpenMP for faster CPU verification
    for (int i = 0; i < num_points; ++i) {
        float min_dist_sq = std::numeric_limits<float>::max();
        int best_cluster_idx = -1;
        float2 my_point = points[i];

        for (int k = 0; k < num_clusters; ++k) {
            float2 centroid_k = centroids[k];
            float dx = my_point.x - centroid_k.x;
            float dy = my_point.y - centroid_k.y;
            float dist_sq = dx * dx + dy * dy;

            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                best_cluster_idx = k;
            }
        }
        assignments_cpu[i] = best_cluster_idx;
    }
}

bool save_assignments(const std::string& filename, const int* assignments, int num_points) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open output file: " << filename << std::endl;
        return false;
    }

    for (int i = 0; i < num_points; ++i) {
        outfile << assignments[i] << "\n";
    }

    if (!outfile.good()) {
         std::cerr << "Error writing to output file: " << filename << std::endl;
         return false;
    }

    return true;
}
