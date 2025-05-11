#include "mcl_localization.cuh"
#include <cmath>     // For powf, expf
#include <vector>
#include <numeric>   // For std::iota
#include <algorithm> // For std::fill, std::transform
#include <iostream>  // For std::cout, std::cerr
#include <iomanip>   // For std::setw, std::fixed, std::setprecision

// --- TransitionMatrix Member Functions ---

TransitionMatrix::TransitionMatrix(int n_states) : num_states(n_states) {
    size_t matrix_size_bytes = num_states * num_states * sizeof(float);
    CHECK_CUDA_ERROR(cudaMalloc(&data, matrix_size_bytes));
    CHECK_CUDA_ERROR(cudaMemset(data, 0, matrix_size_bytes)); // Initialize to zero
}

TransitionMatrix::~TransitionMatrix() {
    if (data) {
        cudaFree(data);
        data = nullptr;
    }
}

void TransitionMatrix::copy_to_device(const float* host_matrix_data) {
    if (!host_matrix_data) {
        std::cerr << "Error: Null pointer provided for host_matrix_data in copy_to_device." << std::endl;
        return;
    }
    size_t matrix_size_bytes = num_states * num_states * sizeof(float);
    CHECK_CUDA_ERROR(cudaMemcpy(data, host_matrix_data, matrix_size_bytes, cudaMemcpyHostToDevice));
}

void TransitionMatrix::copy_to_host(float* host_matrix_data) {
    if (!host_matrix_data) {
        std::cerr << "Error: Null pointer provided for host_matrix_data in copy_to_host." << std::endl;
        return;
    }
    size_t matrix_size_bytes = num_states * num_states * sizeof(float);
    CHECK_CUDA_ERROR(cudaMemcpy(host_matrix_data, data, matrix_size_bytes, cudaMemcpyDeviceToHost));
}

void TransitionMatrix::print_matrix(int max_dim) {
    std::vector<float> host_vector(num_states * num_states); // Create a temporary std::vector
    copy_to_host(host_vector.data()); // Use the raw pointer interface

    std::cout << "Transition Matrix (first " << std::min(max_dim, num_states) << "x" << std::min(max_dim, num_states) << "):" << std::endl;
    for (int i = 0; i < std::min(max_dim, num_states); ++i) {
        for (int j = 0; j < std::min(max_dim, num_states); ++j) {
            std::cout << std::fixed << std::setprecision(4) << std::setw(8) << host_vector[i * num_states + j] << " ";
        }
        std::cout << std::endl;
    }
}


// --- CUDA Kernels ---

// Expansion Kernel: C = A * B (here A=B=input_matrix, C=output_matrix)
// Naive matrix multiplication for demonstration.
__global__ void expansion_kernel(const float* a, const float* b, float* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

// Inflation Kernel Part 1: Element-wise power (M_ij = M_ij ^ gamma)
__global__ void inflation_power_kernel(float* matrix, int num_states, float inflation_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_states * num_states) {
        matrix[idx] = powf(matrix[idx], inflation_factor);
    }
}

// Inflation Kernel Part 2: Normalize columns
// Each column j: M_ij = M_ij / sum_k(M_kj)
// This kernel calculates column sums. A subsequent kernel or host logic would do the division.
// For simplicity and to avoid complex reductions in this example, we'll do a version
// where normalization is done after copying sums back to host, or use a simpler per-column reduction.
// A more advanced version would use shared memory for parallel reduction.
__global__ void inflation_column_sum_kernel(const float* matrix, float* column_sums, int num_states) {
    // Each block processes one column
    int col = blockIdx.x;
    if (col >= num_states) return;

    __shared__ float s_col_sum[256]; // Assuming blockDim.x <= 256
    int tid = threadIdx.x;
    
    s_col_sum[tid] = 0.0f;

    // Sum elements in the column assigned to this block
    for (int row = tid; row < num_states; row += blockDim.x) {
        s_col_sum[tid] += matrix[row * num_states + col];
    }
    __syncthreads();

    // Reduce sum in shared memory (assuming blockDim.x is a power of 2)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_col_sum[tid] += s_col_sum[tid + s];
        }
        __syncthreads();
    }

    // First thread in block writes the sum
    if (tid == 0) {
        column_sums[col] = s_col_sum[0];
    }
}

__global__ void inflation_normalize_kernel(float* matrix, const float* column_sums, int num_states) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_states && col < num_states) {
        if (column_sums[col] > 1e-9f) { // Avoid division by zero
            matrix[row * num_states + col] /= column_sums[col];
        } else if (matrix[row * num_states + col] != 0.0f) {
            // If sum is zero but element is not, this is odd.
            // For safety, could set to 0 or 1/N. Here, set to 0.
            matrix[row * num_states + col] = 0.0f;
        }
        // If sum is zero and element is zero, it remains zero.
    }
}


// --- Kernel Launchers ---

void expand_matrix_cuda(const float* input_matrix_dev, float* output_matrix_dev, int num_states) {
    dim3 threads_per_block(16, 16);
    dim3 num_blocks((num_states + threads_per_block.x - 1) / threads_per_block.x,
                    (num_states + threads_per_block.y - 1) / threads_per_block.y);
    
    expansion_kernel<<<num_blocks, threads_per_block>>>(input_matrix_dev, input_matrix_dev, output_matrix_dev, num_states);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

void inflate_matrix_cuda(float* matrix_dev, int num_states, float inflation_factor) {
    // Part 1: Element-wise power
    int total_elements = num_states * num_states;
    int threads_per_block_power = 256;
    int num_blocks_power = (total_elements + threads_per_block_power - 1) / threads_per_block_power;
    
    inflation_power_kernel<<<num_blocks_power, threads_per_block_power>>>(matrix_dev, num_states, inflation_factor);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Part 2: Normalize columns
    float* d_column_sums;
    CHECK_CUDA_ERROR(cudaMalloc(&d_column_sums, num_states * sizeof(float)));

    int threads_per_block_sum = 256; // Max threads for shared memory reduction
    int num_blocks_sum = num_states; // One block per column
                                     // Ensure num_blocks_sum doesn't exceed device limits if num_states is huge
    
    inflation_column_sum_kernel<<<num_blocks_sum, threads_per_block_sum>>>(matrix_dev, d_column_sums, num_states);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    dim3 threads_per_block_norm(16, 16);
    dim3 num_blocks_norm((num_states + threads_per_block_norm.x - 1) / threads_per_block_norm.x,
                         (num_states + threads_per_block_norm.y - 1) / threads_per_block_norm.y);

    inflation_normalize_kernel<<<num_blocks_norm, threads_per_block_norm>>>(matrix_dev, d_column_sums, num_states);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaFree(d_column_sums));
}


// --- Host-side Interface Functions ---

void initialize_synthetic_grid_world(TransitionMatrix& matrix, int grid_dim) {
    int num_states = matrix.num_states;
    if (num_states != grid_dim * grid_dim) {
        std::cerr << "Error: num_states must equal grid_dim * grid_dim." << std::endl;
        return;
    }

    std::vector<float> host_matrix(num_states * num_states, 0.0f);
    float total_sum_debug = 0.0f;

    // Create a simple connectivity: higher probability to stay or move to adjacent cells
    for (int r1 = 0; r1 < grid_dim; ++r1) {
        for (int c1 = 0; c1 < grid_dim; ++c1) {
            int state1_idx = r1 * grid_dim + c1;
            float column_sum = 0.0f;

            for (int r2 = 0; r2 < grid_dim; ++r2) {
                for (int c2 = 0; c2 < grid_dim; ++c2) {
                    int state2_idx = r2 * grid_dim + c2;
                    
                    int manhattan_dist = std::abs(r1 - r2) + std::abs(c1 - c2);
                    float prob = 0.0f;
                    if (manhattan_dist == 0) prob = 0.5f; // Stay
                    else if (manhattan_dist == 1) prob = 0.1f; // Adjacent
                    else if (manhattan_dist == 2) prob = 0.02f; // Diagonal or 2 steps away
                    // else prob is 0

                    // MCL typically works with symmetric matrices initially, or A^T * A
                    // For simplicity, let's make it symmetric for now.
                    // The matrix M_ij is prob of transitioning from j to i.
                    // So, host_matrix[state2_idx * num_states + state1_idx] is transition from state1 to state2
                    host_matrix[state2_idx * num_states + state1_idx] = prob; 
                    column_sum += prob;
                }
            }
            // Normalize column state1_idx
            if (column_sum > 1e-9f) {
                for (int r2 = 0; r2 < grid_dim; ++r2) {
                    for (int c2 = 0; c2 < grid_dim; ++c2) {
                         int state2_idx = r2 * grid_dim + c2;
                         host_matrix[state2_idx * num_states + state1_idx] /= column_sum;
                    }
                }
            }
             float current_col_sum_check = 0.0f;
             for(int i=0; i < num_states; ++i) current_col_sum_check += host_matrix[i * num_states + state1_idx];
             total_sum_debug += current_col_sum_check;
        }
    }
    // std::cout << "DEBUG: Total sum of column sums after normalization: " << total_sum_debug << std::endl;
    // std::cout << "DEBUG: Expected sum (num_states): " << num_states << std::endl;


    matrix.copy_to_device(host_matrix.data()); // Pass raw pointer from std::vector
}


void mcl_iteration_cuda(TransitionMatrix& matrix, float inflation_factor) {
    // The MCL algorithm typically involves:
    // 1. Expansion: M = M * M (or M = A * M if A is the original adjacency)
    // 2. Inflation: M_ij = (M_ij^gamma) / sum_k(M_kj^gamma)

    // Temporary matrix for expansion result
    float* d_temp_matrix;
    size_t matrix_size_bytes = matrix.num_states * matrix.num_states * sizeof(float);
    CHECK_CUDA_ERROR(cudaMalloc(&d_temp_matrix, matrix_size_bytes));

    // 1. Expansion
    expand_matrix_cuda(matrix.data, d_temp_matrix, matrix.num_states);
    
    // Copy result back to original matrix.data for inflation
    CHECK_CUDA_ERROR(cudaMemcpy(matrix.data, d_temp_matrix, matrix_size_bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA_ERROR(cudaFree(d_temp_matrix));

    // 2. Inflation (operates in-place on matrix.data)
    inflate_matrix_cuda(matrix.data, matrix.num_states, inflation_factor);
}

std::vector<State> extract_clusters_from_probabilities(const TransitionMatrix& matrix, float probability_threshold, int grid_dim) {
    std::vector<float> host_vector(matrix.num_states * matrix.num_states); // Create a temporary std::vector
    // const_cast is okay here as copy_to_host reads from device and writes to host_vector.data(), not modifying matrix's device data.
    const_cast<TransitionMatrix&>(matrix).copy_to_host(host_vector.data()); 

    std::vector<State> clusters;
    int num_states = matrix.num_states;

    // In MCL, after convergence, the matrix columns represent attractors.
    // A simple way to interpret for localization: find states (rows) with high probability in *any* column.
    // Or, if we consider the matrix as P(current_state | initial_uniform_belief_over_one_state),
    // then each column represents the probability distribution if the robot started at that column's state.
    // For a particle filter like interpretation, we might have a single vector of probabilities.
    // Here, let's assume the matrix itself represents P(state_i | state_j) after iterations.
    // We need a way to get a single probability distribution over states.
    // One approach: if the process started with uniform probability, this would be M^k * (1/N, ..., 1/N)^T.
    // Or, sum probabilities across rows or columns if the matrix is symmetric and represents affinities.

    // For this example, let's assume each column j represents a potential cluster attractor.
    // We will find states i that have a high P(i|j) for any j, and also P(j|i) is high (symmetry).
    // A simpler interpretation for "localization": assume the diagonal M_ii represents P(being in state i).
    // This is not strictly MCL interpretation but a simplification for this example.
    
    // Let's iterate through the diagonal elements as a proxy for state probabilities.
    // This is a simplification. A proper MCL clustering would identify columns that are non-zero
    // and group states based on which attractor column they belong to.
    for (int i = 0; i < num_states; ++i) {
        // Consider the diagonal element M_ii as the "belief" or "attraction strength" of state i
        float prob = host_vector[i * num_states + i]; 
        if (prob > probability_threshold) {
            State s;
            s.x = static_cast<float>(i % grid_dim);
            s.y = static_cast<float>(i / grid_dim);
            s.probability = prob;
            clusters.push_back(s);
        }
    }
    
    // A more MCL-like approach:
    // For each column `j` (potential attractor):
    //   If column `j` is "strong" (e.g., sum of its elements or its diagonal is high):
    //     Identify all states `i` that are strongly attracted to `j` (M_ij is high).
    // This is more complex to implement briefly here. The diagonal approach is a placeholder.

    if (clusters.empty() && num_states > 0) {
         // If no state meets threshold, maybe return the one with max probability
        float max_prob = -1.0f;
        int max_idx = -1;
        for(int i=0; i < num_states; ++i) {
            if(host_vector[i*num_states + i] > max_prob){
                max_prob = host_vector[i*num_states + i];
                max_idx = i;
            }
        }
        if(max_idx != -1){
            State s;
            s.x = static_cast<float>(max_idx % grid_dim);
            s.y = static_cast<float>(max_idx / grid_dim);
            s.probability = max_prob;
            clusters.push_back(s);
            std::cout << "Note: No state above threshold, returning max probability state." << std::endl;
        }
    }


    return clusters;
}
