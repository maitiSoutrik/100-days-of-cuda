#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>

// Simple CUDA Error Checking Macro
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

// Define the Axis-Aligned Bounding Box (AABB) structure
typedef struct {
    float3 min;
    float3 max;
} AABB;

// Device function to check overlap on a single axis
__device__ inline bool checkOverlap(float min1, float max1, float min2, float max2) {
    return max1 >= min2 && max2 >= min1;
}

// CUDA kernel to check for collisions between pairs of AABBs
__global__ void checkAABBCollisionKernel(const AABB* boxes, bool* collisionResults, int numBoxes) {
    // Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Total number of unique pairs (n * (n - 1) / 2)
    // We map the 1D thread index to a 2D pair index (i, j) where i < j
    // This mapping is a bit complex, derived from triangular numbers.
    // Instead, let's use a simpler grid-stride loop approach if needed,
    // or a direct mapping if the grid size matches the number of pairs.
    // For simplicity here, let's assume gridDim.x * blockDim.x is exactly
    // the number of pairs we need to check, which is N*(N-1)/2.
    // A more robust way is needed for arbitrary N.

    // Calculate pair indices (i, j) from the linear thread index 'idx'
    // This ensures each thread checks a unique pair (i, j) where i < j.
    // Formula derivation: We want to find i, j such that idx = i * numBoxes + j - (i * (i + 1) / 2) - (i + 1) (approx)
    // A simpler way: iterate through pairs in the kernel if grid size is smaller
    // A robust mapping from 1D index `idx` to pair (i, j) with i < j:
    // Solve `idx = i * n - i*(i+1)/2 + (j - (i+1))` for i and j.
    // Find largest `i` such that `i*n - i*(i+1)/2 <= idx`.
    // Approximate `i` using quadratic formula on `i^2 - (2n-1)i + 2*idx = 0`.
    // `i = floor(( (2n-1) - sqrt((2n-1)^2 - 8*idx) ) / 2)`
    // `j = idx - (i*n - i*(i+1)/2) + i + 1`

    int n = numBoxes;
    if (idx >= (long long)n * (n - 1) / 2) {
        return; // Out of bounds for pair indices
    }

    // Robust mapping from idx to (i, j) pair with i < j
    int i = floor(((2.0 * n - 1.0) - sqrt(pow(2.0 * n - 1.0, 2.0) - 8.0 * idx)) / 2.0);
    int j = idx - (long long)i * n + ((long long)i * (i + 1)) / 2 + i + 1;


    if (i >= n || j >= n || i >= j) {
       // This shouldn't happen with the correct formula and bounds check, but safeguard.
       // printf("Thread %d: Invalid pair (%d, %d)\n", idx, i, j); // Debugging
       return;
    }


    const AABB box1 = boxes[i];
    const AABB box2 = boxes[j];

    // Check for overlap on all three axes (X, Y, Z)
    bool collisionX = checkOverlap(box1.min.x, box1.max.x, box2.min.x, box2.max.x);
    bool collisionY = checkOverlap(box1.min.y, box1.max.y, box2.min.y, box2.max.y);
    bool collisionZ = checkOverlap(box1.min.z, box1.max.z, box2.min.z, box2.max.z);

    // Collision occurs if there is overlap on all axes
    collisionResults[idx] = collisionX && collisionY && collisionZ;
}

// Host function to manage GPU execution
void checkAABBCollisionGPU(const AABB* h_boxes, bool* h_collisionResults, int numBoxes) {
    AABB* d_boxes;
    bool* d_collisionResults;
    long long numPairs = (long long)numBoxes * (numBoxes - 1) / 2;

    if (numPairs == 0) return; // No pairs to check

    size_t boxesBytes = numBoxes * sizeof(AABB);
    size_t resultsBytes = numPairs * sizeof(bool);

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_boxes, boxesBytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_collisionResults, resultsBytes));

    // Copy AABB data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_boxes, h_boxes, boxesBytes, cudaMemcpyHostToDevice));

    // Define kernel launch parameters
    int threadsPerBlock = 256;
    // Ensure sufficient blocks to cover all pairs
    int blocksPerGrid = (numPairs + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    checkAABBCollisionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_boxes, d_collisionResults, numBoxes);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for kernel completion

    // Copy results from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_collisionResults, d_collisionResults, resultsBytes, cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_boxes));
    CHECK_CUDA_ERROR(cudaFree(d_collisionResults));
}

// Host function for CPU collision check (for verification)
void checkAABBCollisionCPU(const AABB* h_boxes, bool* h_collisionResultsCPU, int numBoxes) {
    long long pairIndex = 0;
    for (int i = 0; i < numBoxes; ++i) {
        for (int j = i + 1; j < numBoxes; ++j) {
            const AABB box1 = h_boxes[i];
            const AABB box2 = h_boxes[j];

            bool collisionX = (box1.max.x >= box2.min.x) && (box2.max.x >= box1.min.x);
            bool collisionY = (box1.max.y >= box2.min.y) && (box2.max.y >= box1.min.y);
            bool collisionZ = (box1.max.z >= box2.min.z) && (box2.max.z >= box1.min.z);

            h_collisionResultsCPU[pairIndex++] = collisionX && collisionY && collisionZ;
        }
    }
}

// Helper to generate random float
float randFloat(float low, float high) {
    return low + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (high - low)));
}

int main() {
    srand(time(0)); // Seed random number generator

    int numBoxes = 2000; // Number of AABBs - Increased size
    printf("Starting AABB Collision Check for %d boxes.\n", numBoxes);

    long long numPairs = (long long)numBoxes * (numBoxes - 1) / 2;
    printf("Total unique pairs to check: %lld\n", numPairs);

    if (numPairs == 0) {
        printf("No pairs to check.\n");
        return 0;
    }


    // Allocate host memory
    AABB* h_boxes = (AABB*)malloc(numBoxes * sizeof(AABB));
    bool* h_collisionResultsGPU = (bool*)malloc(numPairs * sizeof(bool));
    bool* h_collisionResultsCPU = (bool*)malloc(numPairs * sizeof(bool));

    if (!h_boxes || !h_collisionResultsGPU || !h_collisionResultsCPU) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        free(h_boxes); free(h_collisionResultsGPU); free(h_collisionResultsCPU);
        return EXIT_FAILURE;
    }


    // Initialize AABB data (randomly generated)
    float boxSizeMean = 1.0f;
    float boxSizeStdDev = 0.2f;
    float spaceSize = 10.0f; // Define the space where boxes exist

    printf("Generating %d random AABBs...\n", numBoxes);
    for (int i = 0; i < numBoxes; ++i) {
        float sizeX = fmaxf(0.1f, randFloat(boxSizeMean - boxSizeStdDev, boxSizeMean + boxSizeStdDev));
        float sizeY = fmaxf(0.1f, randFloat(boxSizeMean - boxSizeStdDev, boxSizeMean + boxSizeStdDev));
        float sizeZ = fmaxf(0.1f, randFloat(boxSizeMean - boxSizeStdDev, boxSizeMean + boxSizeStdDev));

        float centerX = randFloat(-spaceSize / 2.0f, spaceSize / 2.0f);
        float centerY = randFloat(-spaceSize / 2.0f, spaceSize / 2.0f);
        float centerZ = randFloat(-spaceSize / 2.0f, spaceSize / 2.0f);

        h_boxes[i].min = make_float3(centerX - sizeX / 2.0f, centerY - sizeY / 2.0f, centerZ - sizeZ / 2.0f);
        h_boxes[i].max = make_float3(centerX + sizeX / 2.0f, centerY + sizeY / 2.0f, centerZ + sizeZ / 2.0f);
    }
    printf("AABB generation complete.\n");

    // Run GPU version
    printf("Running GPU collision check...\n");
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start));

    checkAABBCollisionGPU(h_boxes, h_collisionResultsGPU, numBoxes);

    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float gpuMillis = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpuMillis, start, stop));
    printf("GPU execution time: %.3f ms\n", gpuMillis);
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));


    // Run CPU version for verification
    printf("Running CPU collision check for verification...\n");
    clock_t cpu_start = clock();
    checkAABBCollisionCPU(h_boxes, h_collisionResultsCPU, numBoxes);
    clock_t cpu_end = clock();
    double cpuMillis = ((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC) * 1000.0;
    printf("CPU execution time: %.3f ms\n", cpuMillis);


    // Verify results
    printf("Verifying results...\n");
    long long mismatches = 0;
    long long gpuCollisions = 0;
    long long cpuCollisions = 0;
    for (long long k = 0; k < numPairs; ++k) {
        if (h_collisionResultsGPU[k]) gpuCollisions++;
        if (h_collisionResultsCPU[k]) cpuCollisions++;
        if (h_collisionResultsGPU[k] != h_collisionResultsCPU[k]) {
            mismatches++;
            // Find the (i, j) pair corresponding to index k for debugging
             if (mismatches < 10) { // Print first few mismatches
                int n = numBoxes;
                int i = floor(((2.0 * n - 1.0) - sqrt(pow(2.0 * n - 1.0, 2.0) - 8.0 * k)) / 2.0);
                int j = k - (long long)i * n + ((long long)i * (i + 1)) / 2 + i + 1;
                printf("Mismatch at pair index %lld (boxes %d and %d): GPU=%d, CPU=%d\n",
                       k, i, j, h_collisionResultsGPU[k], h_collisionResultsCPU[k]);
             }
        }
    }

    printf("Verification complete.\n");
    printf("Total Collisions (GPU): %lld / %lld\n", gpuCollisions, numPairs);
    printf("Total Collisions (CPU): %lld / %lld\n", cpuCollisions, numPairs);

    if (mismatches == 0) {
        printf("SUCCESS: GPU and CPU results match!\n");
    } else {
        printf("FAILURE: %lld mismatches found between GPU and CPU results!\n", mismatches);
    }

    // Free host memory
    free(h_boxes);
    free(h_collisionResultsGPU);
    free(h_collisionResultsCPU);

    printf("Day 48 execution finished.\n");
    return 0;
}
