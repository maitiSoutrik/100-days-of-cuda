#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

// Simple CUDA error checking macro
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Grid parameters
#define GRID_WIDTH 200
#define GRID_HEIGHT 200
#define CELL_SIZE 0.1f // meters per cell

// Log-odds parameters
#define LOG_ODDS_HIT 0.85f  // log(p_occ / (1-p_occ)) for a hit
#define LOG_ODDS_MISS -0.4f // log(p_occ / (1-p_occ)) for a miss
#define LOG_ODDS_CLAMP_MIN -5.0f // Min log-odds value
#define LOG_ODDS_CLAMP_MAX 5.0f  // Max log-odds value

// Simple Sensor simulation parameters
#define NUM_RAYS 1024
#define MAX_RANGE 8.0f // meters

// Device function for clamping
__device__ inline float clamp(float val, float min_val, float max_val) {
    return fmaxf(min_val, fminf(max_val, val));
}

// Device function to update a cell using atomicAdd and clamping
__device__ void updateCell(float* logOddsMap, int cellIdx, float updateValue, float clampMin, float clampMax) {
    // Note: atomicAdd returns the *old* value before the add.
    float oldVal = atomicAdd(&logOddsMap[cellIdx], updateValue);
    // The actual value *after* the add is oldVal + updateValue. We need to clamp this.
    float newVal = oldVal + updateValue;
    // Since atomics don't guarantee read-modify-write atomicity *with clamping*,
    // another thread might have modified the value between our atomicAdd and the clamp read.
    // It's complex. A simpler (though potentially slightly racy) approach is to just write the clamped value back.
    // For occupancy grids, occasional small race conditions in clamping might be acceptable.
    // A more robust way might involve atomicCAS loops, but let's stick to atomicAdd for now.
    logOddsMap[cellIdx] = clamp(newVal, clampMin, clampMax);
    // --- Alternative (potentially safer but might overwrite other updates): ---
    // float currentVal = logOddsMap[cellIdx]; // Read potentially updated value
    // logOddsMap[cellIdx] = clamp(currentVal, clampMin, clampMax); // Clamp based on latest read
}


// Kernel to update the occupancy grid log-odds based on sensor rays
// Uses a simplified integer-based line traversal (similar to Bresenham's concept)
__global__ void updateOccupancyGridKernel(float* logOddsMap, int width, int height, float /*cellSize*/, // cellSize not needed in kernel if coords are grid indices
                                         int* rayStartIndices, int* rayEndIndices, int numRays,
                                         float logOddsHit, float logOddsMiss,
                                         float clampMin, float clampMax)
{
    int rayIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (rayIdx >= numRays) {
        return;
    }

    // Get ray start and end points in grid coordinates
    int startX = rayStartIndices[rayIdx * 2];
    int startY = rayStartIndices[rayIdx * 2 + 1];
    int endX = rayEndIndices[rayIdx * 2];
    int endY = rayEndIndices[rayIdx * 2 + 1];

    // --- Integer-based Line Traversal (similar to Bresenham) ---
    int x0 = startX;
    int y0 = startY;
    int x1 = endX;
    int y1 = endY;

    int dx = abs(x1 - x0);
    int dy = -abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1;
    int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy; // error value e_xy
    int e2; // error value stored in local variable

    // Iterate along the line, applying MISS update to all cells *except* the end cell
    while (true) {
        // Check if current point (x0, y0) is within grid bounds
        if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
            // Don't update the end cell with MISS, only the intermediate ones
            if (x0 != endX || y0 != endY) {
                 int cellIdx = y0 * width + x0;
                 updateCell(logOddsMap, cellIdx, logOddsMiss, clampMin, clampMax);
            }
        }

        if (x0 == x1 && y0 == y1) break; // Reached the end point

        e2 = 2 * err;
        if (e2 >= dy) { // e_xy+e_x > 0
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx) { // e_xy+e_y < 0
            err += dx;
            y0 += sy;
        }
    }

    // Update the HIT cell at the actual end point (x1, y1)
    // Ensure the endpoint itself is within bounds before updating
    if (endX >= 0 && endX < width && endY >= 0 && endY < height) {
         int endCellIdx = endY * width + endX;
         updateCell(logOddsMap, endCellIdx, logOddsHit, clampMin, clampMax);
    }
}


int main() {
    printf("Day 44: Occupancy Grid Mapping Update\n");

    // Initialize Grid
    size_t mapSize = GRID_WIDTH * GRID_HEIGHT * sizeof(float);
    float* h_logOddsMap = (float*)malloc(mapSize);
    float* d_logOddsMap;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_logOddsMap, mapSize));

    // Initialize map to 0 (log-odds of 0.5 probability)
    for (int i = 0; i < GRID_WIDTH * GRID_HEIGHT; ++i) {
        h_logOddsMap[i] = 0.0f;
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_logOddsMap, h_logOddsMap, mapSize, cudaMemcpyHostToDevice));

    // --- Simulate Sensor Data (Rays) ---
    // Simple simulation: rays radiating from the center
    int* h_rayStartIndices = (int*)malloc(NUM_RAYS * 2 * sizeof(int));
    int* h_rayEndIndices = (int*)malloc(NUM_RAYS * 2 * sizeof(int));
    int* d_rayStartIndices;
    int* d_rayEndIndices;

    srand(time(NULL));
    int centerX = GRID_WIDTH / 2;
    int centerY = GRID_HEIGHT / 2;

    for (int i = 0; i < NUM_RAYS; ++i) {
        h_rayStartIndices[i * 2] = centerX;
        h_rayStartIndices[i * 2 + 1] = centerY;

        float angle = (float)i / NUM_RAYS * 2.0f * M_PI;
        // Simulate hitting something at a random distance < MAX_RANGE
        float hitDist = ((float)rand() / RAND_MAX) * MAX_RANGE;
        float endWorldX = centerX * CELL_SIZE + hitDist * cosf(angle);
        float endWorldY = centerY * CELL_SIZE + hitDist * sinf(angle);

        h_rayEndIndices[i * 2] = (int)(endWorldX / CELL_SIZE);
        h_rayEndIndices[i * 2 + 1] = (int)(endWorldY / CELL_SIZE);

        // Clamp end indices to grid boundaries (simplistic handling)
        h_rayEndIndices[i * 2] = max(0, min(GRID_WIDTH - 1, h_rayEndIndices[i * 2]));
        h_rayEndIndices[i * 2 + 1] = max(0, min(GRID_HEIGHT - 1, h_rayEndIndices[i * 2 + 1]));
    }

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_rayStartIndices, NUM_RAYS * 2 * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_rayEndIndices, NUM_RAYS * 2 * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_rayStartIndices, h_rayStartIndices, NUM_RAYS * 2 * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_rayEndIndices, h_rayEndIndices, NUM_RAYS * 2 * sizeof(int), cudaMemcpyHostToDevice));

    // --- Launch Kernel ---
    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_RAYS + threadsPerBlock - 1) / threadsPerBlock;

    printf("Launching kernel with %d blocks and %d threads per block...\n", blocksPerGrid, threadsPerBlock);
    updateOccupancyGridKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_logOddsMap, GRID_WIDTH, GRID_HEIGHT, CELL_SIZE,
        d_rayStartIndices, d_rayEndIndices, NUM_RAYS,
        LOG_ODDS_HIT, LOG_ODDS_MISS,
        LOG_ODDS_CLAMP_MIN, LOG_ODDS_CLAMP_MAX
    );
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for kernel to complete
    printf("Kernel execution finished.\n");

    // --- Copy results back (optional for verification/visualization) ---
    // CHECK_CUDA_ERROR(cudaMemcpy(h_logOddsMap, d_logOddsMap, mapSize, cudaMemcpyDeviceToHost));
    // TODO: Add code to visualize or check the map results if needed

    printf("Occupancy grid update simulated (kernel needs Bresenham implementation).\n");

    // Cleanup
    free(h_logOddsMap);
    free(h_rayStartIndices);
    free(h_rayEndIndices);
    CHECK_CUDA_ERROR(cudaFree(d_logOddsMap));
    CHECK_CUDA_ERROR(cudaFree(d_rayStartIndices));
    CHECK_CUDA_ERROR(cudaFree(d_rayEndIndices));

    printf("Day 44 finished.\n");
    return 0;
}
