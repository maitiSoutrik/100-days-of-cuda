#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Constants for option pricing
#define BLOCK_SIZE 256

// Option parameters structure
typedef struct {
    float S0;        // Initial stock price
    float K;         // Strike price
    float r;         // Risk-free interest rate
    float sigma;     // Volatility
    float T;         // Time to maturity (in years)
    int   optionType; // 0 for call, 1 for put
} OptionData;

// Black-Scholes formula for European options (for comparison)
float blackScholesFormula(const OptionData& option) {
    float d1 = (log(option.S0 / option.K) + (option.r + 0.5f * option.sigma * option.sigma) * option.T) / 
               (option.sigma * sqrt(option.T));
    float d2 = d1 - option.sigma * sqrt(option.T);
    
    float cnd_d1 = 0.5f * (1.0f + erf(d1 / sqrt(2.0f)));
    float cnd_d2 = 0.5f * (1.0f + erf(d2 / sqrt(2.0f)));
    
    if (option.optionType == 0) { // Call option
        return option.S0 * cnd_d1 - option.K * exp(-option.r * option.T) * cnd_d2;
    } else { // Put option
        return option.K * exp(-option.r * option.T) * (1.0f - cnd_d2) - option.S0 * (1.0f - cnd_d1);
    }
}

// CUDA kernel for initializing random number generators
__global__ void setupRandomKernel(curandState *state, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &state[idx]);
}

// CUDA kernel for Monte Carlo simulation of European options
__global__ void monteCarloEuropeanKernel(curandState *state, float *d_results, OptionData option, int paths) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < paths) {
        // Load the random state
        curandState localState = state[idx];
        
        // Generate a random path for the stock price
        float z = curand_normal(&localState);
        float stockPrice = option.S0 * exp((option.r - 0.5f * option.sigma * option.sigma) * option.T + 
                                          option.sigma * sqrt(option.T) * z);
        
        // Calculate the payoff
        float payoff = 0.0f;
        if (option.optionType == 0) { // Call option
            payoff = fmaxf(stockPrice - option.K, 0.0f);
        } else { // Put option
            payoff = fmaxf(option.K - stockPrice, 0.0f);
        }
        
        // Discount the payoff
        d_results[idx] = exp(-option.r * option.T) * payoff;
        
        // Save the random state
        state[idx] = localState;
    }
}

// CUDA kernel for reduction (sum)
__global__ void reduceSum(float *d_results, float *d_sum, int paths) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < paths) ? d_results[i] : 0.0f;
    __syncthreads();
    
    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        d_sum[blockIdx.x] = sdata[0];
    }
}

int main(int argc, char **argv) {
    // Default option parameters
    OptionData option = {
        100.0f,  // S0: Initial stock price
        100.0f,  // K: Strike price
        0.05f,   // r: Risk-free interest rate
        0.2f,    // sigma: Volatility
        1.0f,    // T: Time to maturity (in years)
        0        // optionType: 0 for call, 1 for put
    };
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--S0") == 0 && i + 1 < argc) {
            option.S0 = atof(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--K") == 0 && i + 1 < argc) {
            option.K = atof(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--r") == 0 && i + 1 < argc) {
            option.r = atof(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--sigma") == 0 && i + 1 < argc) {
            option.sigma = atof(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--T") == 0 && i + 1 < argc) {
            option.T = atof(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--type") == 0 && i + 1 < argc) {
            if (strcmp(argv[i + 1], "call") == 0) {
                option.optionType = 0;
            } else if (strcmp(argv[i + 1], "put") == 0) {
                option.optionType = 1;
            }
            i++;
        }
    }
    
    // Number of Monte Carlo simulations
    int numSimulations[] = {10000, 100000, 1000000};
    
    // Calculate the analytical solution using Black-Scholes formula
    float analyticalPrice = blackScholesFormula(option);
    printf("Option Parameters:\n");
    printf("  Initial Price (S0): %.2f\n", option.S0);
    printf("  Strike Price (K): %.2f\n", option.K);
    printf("  Risk-free Rate (r): %.4f\n", option.r);
    printf("  Volatility (sigma): %.4f\n", option.sigma);
    printf("  Time to Maturity (T): %.2f years\n", option.T);
    printf("  Option Type: %s\n", option.optionType == 0 ? "Call" : "Put");
    printf("\nAnalytical Price (Black-Scholes): %.6f\n\n", analyticalPrice);
    
    printf("Monte Carlo Simulation Results:\n");
    printf("%-12s %-15s %-15s %-15s\n", "Simulations", "Option Price", "Abs Error", "Execution Time (ms)");
    
    for (int simIdx = 0; simIdx < 3; simIdx++) {
        int paths = numSimulations[simIdx];
        
        // Allocate memory for random states and results
        curandState *d_states;
        float *d_results, *d_blockSums, *h_blockSums;
        
        cudaMalloc((void **)&d_states, paths * sizeof(curandState));
        cudaMalloc((void **)&d_results, paths * sizeof(float));
        
        // Calculate grid and block dimensions
        int blockSize = BLOCK_SIZE;
        int gridSize = (paths + blockSize - 1) / blockSize;
        
        // Allocate memory for block sums
        cudaMalloc((void **)&d_blockSums, gridSize * sizeof(float));
        h_blockSums = (float *)malloc(gridSize * sizeof(float));
        
        // Create CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // Start timing
        cudaEventRecord(start);
        
        // Initialize random number generators
        setupRandomKernel<<<gridSize, blockSize>>>(d_states, time(NULL));
        
        // Run Monte Carlo simulation
        monteCarloEuropeanKernel<<<gridSize, blockSize>>>(d_states, d_results, option, paths);
        
        // Perform reduction to calculate the sum
        reduceSum<<<gridSize, blockSize>>>(d_results, d_blockSums, paths);
        
        // Copy block sums back to host
        cudaMemcpy(h_blockSums, d_blockSums, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Calculate the final sum on the host
        float sum = 0.0f;
        for (int i = 0; i < gridSize; i++) {
            sum += h_blockSums[i];
        }
        
        // Calculate the option price
        float optionPrice = sum / paths;
        
        // Stop timing
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        // Calculate elapsed time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        // Calculate absolute error
        float absError = fabs(optionPrice - analyticalPrice);
        
        // Print results
        printf("%-12d %-15.6f %-15.6f %-15.2f\n", paths, optionPrice, absError, milliseconds);
        
        // Free memory
        cudaFree(d_states);
        cudaFree(d_results);
        cudaFree(d_blockSums);
        free(h_blockSums);
        
        // Destroy CUDA events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    return 0;
}