#ifndef TVD_LOSS_CUH
#define TVD_LOSS_CUH

#include <vector>

// Error checking macro
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

/**
 * @brief Calculates the Total Variation Distance (TVD) between two discrete probability distributions on the GPU.
 * 
 * The two input vectors are assumed to represent probability mass functions (PMFs),
 * meaning their elements are non-negative and sum to 1.
 * This function does not explicitly check for this condition for performance reasons,
 * so it should be ensured by the caller.
 * 
 * TVD = 0.5 * sum(|P_i - Q_i|)
 * 
 * @param d_p Pointer to the first probability distribution on the device.
 * @param d_q Pointer to the second probability distribution on the device.
 * @param n Number of elements in each distribution.
 * @param d_tvd Pointer to a float on the device where the result (TVD) will be stored.
 */
void calculate_tvd_gpu(const float* d_p, const float* d_q, int n, float* d_tvd);

/**
 * @brief Calculates the Total Variation Distance (TVD) between two discrete probability distributions on the CPU.
 * 
 * @param h_p Host vector representing the first probability distribution.
 * @param h_q Host vector representing the second probability distribution.
 * @return The calculated TVD value.
 */
float calculate_tvd_cpu(const std::vector<float>& h_p, const std::vector<float>& h_q);

#endif // TVD_LOSS_CUH
