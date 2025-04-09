#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <sys/time.h>

// Simulation parameters
#define GRID_WIDTH 512
#define GRID_HEIGHT 512
#define NUM_TIMESTEPS 5000
#define DT 0.25f          // Time step size
#define ALPHA 0.1f        // Diffusion constant
#define LOG_INTERVAL 100  // Log every N steps

// Tile dimensions for shared memory kernel
#define TILE_DIM 16

// Function to check for CUDA errors
void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s failed with error: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Get current time in milliseconds
double getCurrentTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

// Initialize the grid with a heat source in the center
void initializeGrid(float *grid, int width, int height) {
    // Set all cells to 0.0
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            grid[y * width + x] = 0.0f;
        }
    }
    
    // Set a heat source in the center (100.0 degrees)
    int centerX = width / 2;
    int centerY = height / 2;
    int blockSize = width / 10; // 10% of grid width
    
    for (int y = centerY - blockSize/2; y < centerY + blockSize/2; y++) {
        for (int x = centerX - blockSize/2; x < centerX + blockSize/2; x++) {
            grid[y * width + x] = 100.0f;
        }
    }
}

// Basic heat diffusion kernel
__global__ void heat_step_kernel(const float* current_grid, float* next_grid, int width, int height, float dt, float alpha) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;
    
    // Boundary check
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return;
    }
    
    // Handle boundary conditions (fixed temperature at boundaries)
    if (x == 0 || x == width-1 || y == 0 || y == height-1) {
        next_grid[idx] = current_grid[idx];
        return;
    }
    
    // Read center and neighbors
    float center = current_grid[idx];
    float left = current_grid[idx - 1];
    float right = current_grid[idx + 1];
    float up = current_grid[idx - width];
    float down = current_grid[idx + width];
    
    // Calculate Laplacian (second derivative approximation)
    float laplacian = (left + right + up + down - 4.0f * center);
    
    // Update temperature using the heat equation
    next_grid[idx] = center + dt * alpha * laplacian;
}

// Shared memory heat diffusion kernel
__global__ void heat_step_shared_kernel(const float* current_grid, float* next_grid, int width, int height, float dt, float alpha) {
    // Shared memory for tile including halo cells
    __shared__ float tile[TILE_DIM + 2][TILE_DIM + 2];
    
    // Global indices
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;
    
    // Local indices for shared memory
    int tx = threadIdx.x + 1;  // +1 for halo cells
    int ty = threadIdx.y + 1;  // +1 for halo cells
    
    // Load center cell
    if (x < width && y < height) {
        tile[ty][tx] = current_grid[idx];
    }
    
    // Load halo cells
    if (threadIdx.x == 0 && x > 0) {
        // Left halo
        tile[ty][tx-1] = current_grid[idx-1];
    }
    if (threadIdx.x == blockDim.x-1 && x < width-1) {
        // Right halo
        tile[ty][tx+1] = current_grid[idx+1];
    }
    if (threadIdx.y == 0 && y > 0) {
        // Top halo
        tile[ty-1][tx] = current_grid[idx-width];
    }
    if (threadIdx.y == blockDim.y-1 && y < height-1) {
        // Bottom halo
        tile[ty+1][tx] = current_grid[idx+width];
    }
    
    // Synchronize to make sure all data is loaded
    __syncthreads();
    
    // Boundary check
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return;
    }
    
    // Handle boundary conditions (fixed temperature at boundaries)
    if (x == 0 || x == width-1 || y == 0 || y == height-1) {
        next_grid[idx] = tile[ty][tx];
        return;
    }
    
    // Calculate Laplacian using shared memory
    float center = tile[ty][tx];
    float left = tile[ty][tx-1];
    float right = tile[ty][tx+1];
    float up = tile[ty-1][tx];
    float down = tile[ty+1][tx];
    
    float laplacian = (left + right + up + down - 4.0f * center);
    
    // Update temperature using the heat equation
    next_grid[idx] = center + dt * alpha * laplacian;
}

// Reduction kernel to find the sum of all temperatures
__global__ void reduce_sum_kernel(const float* grid, float* result, int size) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < size) ? grid[i] : 0.0f;
    __syncthreads();
    
    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

// Reduction kernel to find the maximum temperature
__global__ void reduce_max_kernel(const float* grid, float* result, int size) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < size) ? grid[i] : 0.0f;
    __syncthreads();
    
    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

int main(int argc, char** argv) {
    // Print device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device name: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Total global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    
    // Print simulation parameters
    printf("\n----- Simulation Parameters -----\n");
    printf("Grid size: %d x %d\n", GRID_WIDTH, GRID_HEIGHT);
    printf("Number of timesteps: %d\n", NUM_TIMESTEPS);
    printf("Time step size (dt): %.4f\n", DT);
    printf("Diffusion constant (alpha): %.4f\n", ALPHA);
    
    // Determine if we should use shared memory based on command line argument
    bool useSharedMemory = false;
    if (argc > 1 && strcmp(argv[1], "shared") == 0) {
        useSharedMemory = true;
        printf("Using shared memory kernel\n");
    } else {
        printf("Using basic kernel (use 'shared' argument to enable shared memory)\n");
    }
    
    // Allocate host memory
    size_t gridSize = GRID_WIDTH * GRID_HEIGHT * sizeof(float);
    float *h_grid = (float*)malloc(gridSize);
    
    // Initialize the grid
    initializeGrid(h_grid, GRID_WIDTH, GRID_HEIGHT);
    
    // Allocate device memory
    float *d_grid_current = NULL;
    float *d_grid_next = NULL;
    checkCudaError(cudaMalloc((void**)&d_grid_current, gridSize), "cudaMalloc d_grid_current");
    checkCudaError(cudaMalloc((void**)&d_grid_next, gridSize), "cudaMalloc d_grid_next");
    
    // Copy initial grid to device
    checkCudaError(cudaMemcpy(d_grid_current, h_grid, gridSize, cudaMemcpyHostToDevice), "cudaMemcpy h_grid to d_grid_current");
    
    // Allocate memory for reduction results
    int threadsPerBlock = 256;
    int numBlocks = (GRID_WIDTH * GRID_HEIGHT + threadsPerBlock - 1) / threadsPerBlock;
    float *d_reduction_output = NULL;
    float *d_final_result = NULL;
    checkCudaError(cudaMalloc((void**)&d_reduction_output, numBlocks * sizeof(float)), "cudaMalloc d_reduction_output");
    checkCudaError(cudaMalloc((void**)&d_final_result, sizeof(float)), "cudaMalloc d_final_result");
    
    // Host memory for reduction results
    float *h_reduction_output = (float*)malloc(numBlocks * sizeof(float));
    float h_final_result;
    
    // Set up kernel launch parameters
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((GRID_WIDTH + blockDim.x - 1) / blockDim.x, 
                 (GRID_HEIGHT + blockDim.y - 1) / blockDim.y);
    
    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    printf("\n----- Starting Simulation -----\n");
    
    // Main simulation loop
    double totalKernelTime = 0.0;
    for (int step = 0; step < NUM_TIMESTEPS; step++) {
        // Start timing
        cudaEventRecord(start);
        
        // Launch the appropriate kernel
        if (useSharedMemory) {
            heat_step_shared_kernel<<<gridDim, blockDim>>>(
                d_grid_current, d_grid_next, GRID_WIDTH, GRID_HEIGHT, DT, ALPHA);
        } else {
            heat_step_kernel<<<gridDim, blockDim>>>(
                d_grid_current, d_grid_next, GRID_WIDTH, GRID_HEIGHT, DT, ALPHA);
        }
        
        // Check for errors
        checkCudaError(cudaGetLastError(), "kernel launch");
        
        // Stop timing
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        // Calculate kernel execution time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalKernelTime += milliseconds;
        
        // Swap pointers for next iteration
        float *temp = d_grid_current;
        d_grid_current = d_grid_next;
        d_grid_next = temp;
        
        // Log statistics every LOG_INTERVAL steps
        if (step % LOG_INTERVAL == 0 || step == NUM_TIMESTEPS - 1) {
            // Calculate average temperature
            reduce_sum_kernel<<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
                d_grid_current, d_reduction_output, GRID_WIDTH * GRID_HEIGHT);
            
            // Second reduction to get final sum
            reduce_sum_kernel<<<1, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
                d_reduction_output, d_final_result, numBlocks);
            
            // Copy sum result back to host
            checkCudaError(cudaMemcpy(&h_final_result, d_final_result, sizeof(float), cudaMemcpyDeviceToHost), 
                          "cudaMemcpy d_final_result to h_final_result");
            
            float avgTemp = h_final_result / (GRID_WIDTH * GRID_HEIGHT);
            
            // Calculate maximum temperature
            reduce_max_kernel<<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
                d_grid_current, d_reduction_output, GRID_WIDTH * GRID_HEIGHT);
            
            // Second reduction to get final max
            reduce_max_kernel<<<1, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
                d_reduction_output, d_final_result, numBlocks);
            
            // Copy max result back to host
            checkCudaError(cudaMemcpy(&h_final_result, d_final_result, sizeof(float), cudaMemcpyDeviceToHost), 
                          "cudaMemcpy d_final_result to h_final_result");
            
            float maxTemp = h_final_result;
            
            // Log the statistics
            printf("Timestep: %d, Avg Temp: %.4f, Max Temp: %.4f, Kernel Time: %.3f ms\n", 
                   step, avgTemp, maxTemp, milliseconds);
        }
    }
    
    // Print performance summary
    printf("\n----- Performance Summary -----\n");
    printf("Total kernel execution time: %.2f ms\n", totalKernelTime);
    printf("Average kernel execution time per step: %.4f ms\n", totalKernelTime / NUM_TIMESTEPS);
    
    // Copy final grid back to host for verification
    checkCudaError(cudaMemcpy(h_grid, d_grid_current, gridSize, cudaMemcpyDeviceToHost), 
                  "cudaMemcpy d_grid_current to h_grid");
    
    // Print a small section of the final grid for verification
    printf("\n----- Final Grid Sample (center region) -----\n");
    int centerX = GRID_WIDTH / 2;
    int centerY = GRID_HEIGHT / 2;
    int sampleSize = 5; // Show a 5x5 sample around the center
    
    for (int y = centerY - sampleSize/2; y <= centerY + sampleSize/2; y++) {
        for (int x = centerX - sampleSize/2; x <= centerX + sampleSize/2; x++) {
            printf("%.2f ", h_grid[y * GRID_WIDTH + x]);
        }
        printf("\n");
    }
    
    // Clean up
    cudaFree(d_grid_current);
    cudaFree(d_grid_next);
    cudaFree(d_reduction_output);
    cudaFree(d_final_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_grid);
    free(h_reduction_output);
    
    return 0;
}
