#include "mcl_localization.cuh"
#include <iostream>
#include <vector>
#include <string> // For std::stof, std::stoi

// Simulation parameters
const int GRID_DIM = 10; // 10x10 grid, so 100 states
const int NUM_STATES = GRID_DIM * GRID_DIM;
const int NUM_ITERATIONS = 10;
const float INFLATION_FACTOR = 2.0f; // Common value for MCL
const float PROBABILITY_THRESHOLD = 0.01f; // For extracting significant states

void print_usage(const char* prog_name) {
    std::cerr << "Usage: " << prog_name << " [grid_dim] [num_iterations] [inflation_factor] [prob_threshold]" << std::endl;
    std::cerr << "Defaults: grid_dim=" << GRID_DIM
              << ", num_iterations=" << NUM_ITERATIONS
              << ", inflation_factor=" << INFLATION_FACTOR
              << ", prob_threshold=" << PROBABILITY_THRESHOLD << std::endl;
}

int main(int argc, char* argv[]) {
    int grid_dim = GRID_DIM;
    int num_iterations = NUM_ITERATIONS;
    float inflation_factor = INFLATION_FACTOR;
    float probability_threshold = PROBABILITY_THRESHOLD;

    if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        print_usage(argv[0]);
        return 0;
    }

    if (argc > 1) grid_dim = std::stoi(argv[1]);
    if (argc > 2) num_iterations = std::stoi(argv[2]);
    if (argc > 3) inflation_factor = std::stof(argv[3]);
    if (argc > 4) probability_threshold = std::stof(argv[4]);

    int num_states = grid_dim * grid_dim;

    std::cout << "Starting MCL Localization Simulation..." << std::endl;
    std::cout << "Grid Dimensions: " << grid_dim << "x" << grid_dim << " (" << num_states << " states)" << std::endl;
    std::cout << "Number of Iterations: " << num_iterations << std::endl;
    std::cout << "Inflation Factor: " << inflation_factor << std::endl;
    std::cout << "Probability Threshold for Cluster Extraction: " << probability_threshold << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;

    // Initialize Transition Matrix
    TransitionMatrix matrix(num_states);
    std::cout << "Initializing synthetic grid world..." << std::endl;
    initialize_synthetic_grid_world(matrix, grid_dim);
    
    std::cout << "Initial Transition Matrix (sample):" << std::endl;
    matrix.print_matrix(std::min(grid_dim, 10)); // Print a small part
    std::cout << "-------------------------------------------------" << std::endl;

    // Run MCL Iterations
    std::cout << "Running MCL iterations..." << std::endl;
    for (int i = 0; i < num_iterations; ++i) {
        mcl_iteration_cuda(matrix, inflation_factor);
        std::cout << "Completed Iteration " << i + 1 << "/" << num_iterations << std::endl;
        if (num_states <= 100) { // Only print for small matrices
             std::cout << "Matrix after iteration " << i+1 << " (sample):" << std::endl;
             matrix.print_matrix(std::min(grid_dim,10));
        }
    }
    std::cout << "-------------------------------------------------" << std::endl;

    std::cout << "Final Transition Matrix (sample):" << std::endl;
    matrix.print_matrix(std::min(grid_dim, 10));
    std::cout << "-------------------------------------------------" << std::endl;

    // Extract and print clusters
    std::cout << "Extracting clusters (states with probability > " << probability_threshold << "):" << std::endl;
    std::vector<State> clusters = extract_clusters_from_probabilities(matrix, probability_threshold, grid_dim);

    if (clusters.empty()) {
        std::cout << "No significant clusters found above the threshold." << std::endl;
    } else {
        std::cout << "Found " << clusters.size() << " significant state(s):" << std::endl;
        for (const auto& state : clusters) {
            std::cout << "  State (" << state.x <&lt ", " << state.y << ") - Probability: " << state.probability << std::endl;
        }
    }
    std::cout << "-------------------------------------------------" << std::endl;
    std::cout << "MCL Localization Simulation Finished." << std::endl;

    return 0;
}
