#ifndef MCL_LOCALIZATION_CUH
#define MCL_LOCALIZATION_CUH

#include <vector>
#include <cstdio> // For fprintf
#include <cstdlib> // For exit
#include "cuda_runtime.h" // For CUDA types and error functions

// Error checking macro
#define CHECK_CUDA_ERROR(err) \
    do { \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Structure to represent a 2D point or state
struct State {
    float x;
    float y;
    float probability; // Probability of this state
};

// Structure for the graph/matrix used in MCL
// For simplicity, we'll use a dense matrix representation for now.
// In a real scenario, a sparse matrix representation (like ELLPACK-R) would be more efficient.
struct TransitionMatrix {
    float* data;       // Pointer to matrix data on device
    int num_states;    // Number of states (matrix is num_states x num_states)
    
    // Constructor and Destructor (CPU-side)
    TransitionMatrix(int n_states);
    ~TransitionMatrix();

    // Use raw pointers for the interface to potentially avoid nvcc issues with std::vector in headers
    void copy_to_device(const float* host_matrix_data);
    void copy_to_host(float* host_matrix_data);
    void print_matrix(int max_dim = 10); // Print a portion of the matrix for debugging
};

// --- Kernel Launchers ---

// Kernel for the Expansion step (Matrix-Matrix Multiplication: M = M * M)
// This is a simplified version; a more robust implementation would use shared memory, etc.
void expand_matrix_cuda(const float* input_matrix, float* output_matrix, int num_states);

// Kernel for the Inflation step (Element-wise power and normalization)
// M_ij = (M_ij ^ gamma) / sum_k(M_kj ^ gamma) for each column j
void inflate_matrix_cuda(float* matrix, int num_states, float inflation_factor);


// --- Host-side Interface Functions ---

// Initializes a transition matrix with synthetic data representing a simple 2D grid world
// For example, higher probabilities for nearby states.
void initialize_synthetic_grid_world(TransitionMatrix& matrix, int grid_dim);

// Performs one iteration of the MCL algorithm (Expansion + Inflation)
void mcl_iteration_cuda(TransitionMatrix& matrix, float inflation_factor);

// Function to extract clusters (simplified: identify states with high probability)
std::vector<State> extract_clusters_from_probabilities(const TransitionMatrix& matrix, float probability_threshold, int grid_dim);

#endif // MCL_LOCALIZATION_CUH
