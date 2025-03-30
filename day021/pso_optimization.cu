#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>

// Custom atomic minimum function for float
__device__ void atomicMinFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int expected;
    
    do {
        expected = old;
        int new_val = __float_as_int(min(__int_as_float(expected), val));
        old = atomicCAS(address_as_int, expected, new_val);
    } while (expected != old);
}

// Error checking macro
#define CHECK_CUDA_ERROR(call) \
{ \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at line %d\n", cudaGetErrorString(error), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// Constants
#define BLOCK_SIZE 256
#define MAX_DIMENSIONS 100

// PSO parameters
typedef struct {
    int numParticles;
    int dimensions;
    int maxIterations;
    float inertiaWeight;
    float cognitiveWeight;
    float socialWeight;
    float minBound;
    float maxBound;
    int functionType;  // 0: Sphere, 1: Rastrigin, 2: Rosenbrock
} PSOParams;

// Particle structure
typedef struct {
    float position[MAX_DIMENSIONS];
    float velocity[MAX_DIMENSIONS];
    float personalBest[MAX_DIMENSIONS];
    float personalBestFitness;
} Particle;

// Function to evaluate the fitness of a position (CPU version)
float evaluateFitness(float* position, int dimensions, int functionType) {
    float fitness = 0.0f;
    
    switch (functionType) {
        case 0: // Sphere function
            for (int i = 0; i < dimensions; i++) {
                fitness += position[i] * position[i];
            }
            break;
            
        case 1: // Rastrigin function
            fitness = 10.0f * dimensions;
            for (int i = 0; i < dimensions; i++) {
                fitness += position[i] * position[i] - 10.0f * cosf(2.0f * M_PI * position[i]);
            }
            break;
            
        case 2: // Rosenbrock function
            for (int i = 0; i < dimensions - 1; i++) {
                float term1 = position[i + 1] - position[i] * position[i];
                float term2 = 1.0f - position[i];
                fitness += 100.0f * term1 * term1 + term2 * term2;
            }
            break;
            
        default:
            fitness = 0.0f;
    }
    
    return fitness;
}

// CUDA kernel for initializing random number generators
__global__ void setupRandomKernel(curandState* state, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &state[idx]);
}

// CUDA kernel for initializing particles
__global__ void initializeParticlesKernel(
    Particle* particles,
    curandState* randStates,
    int numParticles,
    int dimensions,
    float minBound,
    float maxBound
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < numParticles) {
        curandState localState = randStates[idx];
        
        // Initialize position and velocity
        for (int d = 0; d < dimensions; d++) {
            // Random position between minBound and maxBound
            particles[idx].position[d] = minBound + curand_uniform(&localState) * (maxBound - minBound);
            
            // Random velocity between -1 and 1
            particles[idx].velocity[d] = -1.0f + 2.0f * curand_uniform(&localState);
            
            // Initialize personal best to current position
            particles[idx].personalBest[d] = particles[idx].position[d];
        }
        
        // Initialize personal best fitness to infinity (will be updated in the first iteration)
        particles[idx].personalBestFitness = FLT_MAX;
        
        // Save random state
        randStates[idx] = localState;
    }
}

// CUDA kernel for evaluating fitness
__global__ void evaluateFitnessKernel(
    Particle* particles,
    float* fitness,
    int numParticles,
    int dimensions,
    int functionType
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < numParticles) {
        float result = 0.0f;
        
        switch (functionType) {
            case 0: // Sphere function
                for (int i = 0; i < dimensions; i++) {
                    result += particles[idx].position[i] * particles[idx].position[i];
                }
                break;
                
            case 1: // Rastrigin function
                result = 10.0f * dimensions;
                for (int i = 0; i < dimensions; i++) {
                    result += particles[idx].position[i] * particles[idx].position[i] - 
                              10.0f * cosf(2.0f * M_PI * particles[idx].position[i]);
                }
                break;
                
            case 2: // Rosenbrock function
                for (int i = 0; i < dimensions - 1; i++) {
                    float term1 = particles[idx].position[i + 1] - particles[idx].position[i] * particles[idx].position[i];
                    float term2 = 1.0f - particles[idx].position[i];
                    result += 100.0f * term1 * term1 + term2 * term2;
                }
                break;
                
            default:
                result = 0.0f;
        }
        
        fitness[idx] = result;
    }
}

// CUDA kernel for updating personal best
__global__ void updatePersonalBestKernel(
    Particle* particles,
    float* fitness,
    int numParticles
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < numParticles) {
        if (fitness[idx] < particles[idx].personalBestFitness) {
            particles[idx].personalBestFitness = fitness[idx];
            
            // Copy current position to personal best
            for (int d = 0; d < MAX_DIMENSIONS; d++) {
                particles[idx].personalBest[d] = particles[idx].position[d];
            }
        }
    }
}

// Custom atomic minimum function for float (forward declaration)
__device__ void atomicMinFloat(float* address, float val);

// CUDA kernel for finding the global best index
__global__ void findGlobalBestKernel(
    Particle* particles,
    int* globalBestIdx,
    float* globalBestFitness,
    int numParticles
) {
    __shared__ float sharedFitness[BLOCK_SIZE];
    __shared__ int sharedIndices[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    sharedFitness[tid] = FLT_MAX;
    sharedIndices[tid] = -1;
    
    // Load data into shared memory
    if (idx < numParticles) {
        sharedFitness[tid] = particles[idx].personalBestFitness;
        sharedIndices[tid] = idx;
    }
    __syncthreads();
    
    // Perform reduction to find minimum fitness
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sharedFitness[tid] > sharedFitness[tid + s]) {
                sharedFitness[tid] = sharedFitness[tid + s];
                sharedIndices[tid] = sharedIndices[tid + s];
            }
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        atomicMinFloat(globalBestFitness, sharedFitness[0]);
        if (sharedFitness[0] == *globalBestFitness) {
            *globalBestIdx = sharedIndices[0];
        }
    }
}

// Custom atomic minimum function for float (implementation)
__device__ void atomicMinFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int expected;
    
    do {
        expected = old;
        int new_val = __float_as_int(min(__int_as_float(expected), val));
        old = atomicCAS(address_as_int, expected, new_val);
    } while (expected != old);
}

// CUDA kernel for updating particle positions and velocities
__global__ void updateParticlesKernel(
    Particle* particles,
    Particle* globalBest,
    curandState* randStates,
    int numParticles,
    int dimensions,
    float inertiaWeight,
    float cognitiveWeight,
    float socialWeight,
    float minBound,
    float maxBound
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < numParticles) {
        curandState localState = randStates[idx];
        
        // Update velocity and position for each dimension
        for (int d = 0; d < dimensions; d++) {
            // Generate random coefficients
            float r1 = curand_uniform(&localState);
            float r2 = curand_uniform(&localState);
            
            // Update velocity
            particles[idx].velocity[d] = inertiaWeight * particles[idx].velocity[d] +
                                        cognitiveWeight * r1 * (particles[idx].personalBest[d] - particles[idx].position[d]) +
                                        socialWeight * r2 * (globalBest->position[d] - particles[idx].position[d]);
            
            // Update position
            particles[idx].position[d] += particles[idx].velocity[d];
            
            // Clamp position to bounds
            particles[idx].position[d] = fmaxf(minBound, fminf(maxBound, particles[idx].position[d]));
        }
        
        // Save random state
        randStates[idx] = localState;
    }
}

// Function to run PSO on CPU (for comparison)
float runPSOonCPU(PSOParams params) {
    // Allocate memory for particles
    Particle* particles = (Particle*)malloc(params.numParticles * sizeof(Particle));
    
    // Initialize particles
    srand(time(NULL));
    for (int i = 0; i < params.numParticles; i++) {
        for (int d = 0; d < params.dimensions; d++) {
            // Random position between minBound and maxBound
            particles[i].position[d] = params.minBound + ((float)rand() / RAND_MAX) * (params.maxBound - params.minBound);
            
            // Random velocity between -1 and 1
            particles[i].velocity[d] = -1.0f + 2.0f * ((float)rand() / RAND_MAX);
            
            // Initialize personal best to current position
            particles[i].personalBest[d] = particles[i].position[d];
        }
        
        // Initialize personal best fitness
        particles[i].personalBestFitness = evaluateFitness(particles[i].position, params.dimensions, params.functionType);
    }
    
    // Initialize global best
    int globalBestIdx = 0;
    float globalBestFitness = particles[0].personalBestFitness;
    
    // Find initial global best
    for (int i = 1; i < params.numParticles; i++) {
        if (particles[i].personalBestFitness < globalBestFitness) {
            globalBestFitness = particles[i].personalBestFitness;
            globalBestIdx = i;
        }
    }
    
    // Main PSO loop
    for (int iter = 0; iter < params.maxIterations; iter++) {
        // Update particles
        for (int i = 0; i < params.numParticles; i++) {
            // Update velocity and position
            for (int d = 0; d < params.dimensions; d++) {
                float r1 = (float)rand() / RAND_MAX;
                float r2 = (float)rand() / RAND_MAX;
                
                // Update velocity
                particles[i].velocity[d] = params.inertiaWeight * particles[i].velocity[d] +
                                          params.cognitiveWeight * r1 * (particles[i].personalBest[d] - particles[i].position[d]) +
                                          params.socialWeight * r2 * (particles[globalBestIdx].personalBest[d] - particles[i].position[d]);
                
                // Update position
                particles[i].position[d] += particles[i].velocity[d];
                
                // Clamp position to bounds
                particles[i].position[d] = fmaxf(params.minBound, fminf(params.maxBound, particles[i].position[d]));
            }
            
            // Evaluate fitness
            float fitness = evaluateFitness(particles[i].position, params.dimensions, params.functionType);
            
            // Update personal best
            if (fitness < particles[i].personalBestFitness) {
                particles[i].personalBestFitness = fitness;
                
                for (int d = 0; d < params.dimensions; d++) {
                    particles[i].personalBest[d] = particles[i].position[d];
                }
                
                // Update global best
                if (fitness < globalBestFitness) {
                    globalBestFitness = fitness;
                    globalBestIdx = i;
                }
            }
        }
        
        // Print progress every 100 iterations
        if (iter % 100 == 0 || iter == params.maxIterations - 1) {
            printf("CPU - Iteration %d: Best fitness = %.10f\n", iter, globalBestFitness);
        }
    }
    
    // Save the best fitness
    float bestFitness = globalBestFitness;
    
    // Free memory
    free(particles);
    
    return bestFitness;
}

// Function to run PSO on GPU
float runPSOonGPU(PSOParams params) {
    // Allocate host memory for global best
    Particle h_globalBest;
    float h_globalBestFitness = FLT_MAX;
    
    // Allocate device memory
    Particle* d_particles;
    float* d_fitness;
    curandState* d_randStates;
    Particle* d_globalBest;
    int* d_globalBestIdx;
    float* d_globalBestFitness;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_particles, params.numParticles * sizeof(Particle)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_fitness, params.numParticles * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_randStates, params.numParticles * sizeof(curandState)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_globalBest, sizeof(Particle)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_globalBestIdx, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_globalBestFitness, sizeof(float)));
    
    // Calculate grid and block dimensions
    int blockSize = BLOCK_SIZE;
    int gridSize = (params.numParticles + blockSize - 1) / blockSize;
    
    // Initialize random number generators
    setupRandomKernel<<<gridSize, blockSize>>>(d_randStates, time(NULL));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Initialize particles
    initializeParticlesKernel<<<gridSize, blockSize>>>(
        d_particles,
        d_randStates,
        params.numParticles,
        params.dimensions,
        params.minBound,
        params.maxBound
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Initialize global best fitness
    CHECK_CUDA_ERROR(cudaMemcpy(d_globalBestFitness, &h_globalBestFitness, sizeof(float), cudaMemcpyHostToDevice));
    
    // Main PSO loop
    for (int iter = 0; iter < params.maxIterations; iter++) {
        // Evaluate fitness
        evaluateFitnessKernel<<<gridSize, blockSize>>>(
            d_particles,
            d_fitness,
            params.numParticles,
            params.dimensions,
            params.functionType
        );
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Update personal best
        updatePersonalBestKernel<<<gridSize, blockSize>>>(
            d_particles,
            d_fitness,
            params.numParticles
        );
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Find global best
        int h_globalBestIdx = 0;
        CHECK_CUDA_ERROR(cudaMemcpy(d_globalBestIdx, &h_globalBestIdx, sizeof(int), cudaMemcpyHostToDevice));
        
        findGlobalBestKernel<<<gridSize, blockSize>>>(
            d_particles,
            d_globalBestIdx,
            d_globalBestFitness,
            params.numParticles
        );
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Copy global best index and fitness back to host
        CHECK_CUDA_ERROR(cudaMemcpy(&h_globalBestIdx, d_globalBestIdx, sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(&h_globalBestFitness, d_globalBestFitness, sizeof(float), cudaMemcpyDeviceToHost));
        
        // Copy global best particle to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_globalBest, &d_particles[h_globalBestIdx], sizeof(Particle), cudaMemcpyDeviceToDevice));
        
        // Update particles
        updateParticlesKernel<<<gridSize, blockSize>>>(
            d_particles,
            d_globalBest,
            d_randStates,
            params.numParticles,
            params.dimensions,
            params.inertiaWeight,
            params.cognitiveWeight,
            params.socialWeight,
            params.minBound,
            params.maxBound
        );
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Print progress every 100 iterations
        if (iter % 100 == 0 || iter == params.maxIterations - 1) {
            printf("GPU - Iteration %d: Best fitness = %.10f\n", iter, h_globalBestFitness);
        }
    }
    
    // Copy global best back to host
    CHECK_CUDA_ERROR(cudaMemcpy(&h_globalBest, d_globalBest, sizeof(Particle), cudaMemcpyDeviceToHost));
    
    // Print best solution
    printf("\nBest solution found:\n");
    printf("Fitness: %.10f\n", h_globalBestFitness);
    printf("Position: [");
    for (int d = 0; d < params.dimensions; d++) {
        printf("%.6f", h_globalBest.position[d]);
        if (d < params.dimensions - 1) {
            printf(", ");
        }
    }
    printf("]\n");
    
    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_particles));
    CHECK_CUDA_ERROR(cudaFree(d_fitness));
    CHECK_CUDA_ERROR(cudaFree(d_randStates));
    CHECK_CUDA_ERROR(cudaFree(d_globalBest));
    CHECK_CUDA_ERROR(cudaFree(d_globalBestIdx));
    CHECK_CUDA_ERROR(cudaFree(d_globalBestFitness));
    
    return h_globalBestFitness;
}

// Function to get function name from type
const char* getFunctionName(int functionType) {
    switch (functionType) {
        case 0: return "Sphere";
        case 1: return "Rastrigin";
        case 2: return "Rosenbrock";
        default: return "Unknown";
    }
}

int main(int argc, char** argv) {
    // Default PSO parameters
    PSOParams params = {
        1024,       // numParticles
        10,         // dimensions
        1000,       // maxIterations
        0.729f,     // inertiaWeight (Clerc's constriction coefficient)
        1.49445f,   // cognitiveWeight
        1.49445f,   // socialWeight
        -100.0f,    // minBound
        100.0f,     // maxBound
        0           // functionType (0: Sphere)
    };
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--particles") == 0 && i + 1 < argc) {
            params.numParticles = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--dimensions") == 0 && i + 1 < argc) {
            params.dimensions = atoi(argv[i + 1]);
            if (params.dimensions > MAX_DIMENSIONS) {
                printf("Warning: Maximum dimensions is %d. Setting dimensions to %d.\n", 
                       MAX_DIMENSIONS, MAX_DIMENSIONS);
                params.dimensions = MAX_DIMENSIONS;
            }
            i++;
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            params.maxIterations = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--inertia") == 0 && i + 1 < argc) {
            params.inertiaWeight = atof(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--cognitive") == 0 && i + 1 < argc) {
            params.cognitiveWeight = atof(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--social") == 0 && i + 1 < argc) {
            params.socialWeight = atof(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--min") == 0 && i + 1 < argc) {
            params.minBound = atof(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--max") == 0 && i + 1 < argc) {
            params.maxBound = atof(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--function") == 0 && i + 1 < argc) {
            if (strcmp(argv[i + 1], "sphere") == 0) {
                params.functionType = 0;
            } else if (strcmp(argv[i + 1], "rastrigin") == 0) {
                params.functionType = 1;
            } else if (strcmp(argv[i + 1], "rosenbrock") == 0) {
                params.functionType = 2;
            }
            i++;
        }
    }
    
    // Print parameters
    printf("Particle Swarm Optimization (PSO)\n");
    printf("=================================\n");
    printf("Function: %s\n", getFunctionName(params.functionType));
    printf("Dimensions: %d\n", params.dimensions);
    printf("Particles: %d\n", params.numParticles);
    printf("Iterations: %d\n", params.maxIterations);
    printf("Inertia Weight: %.4f\n", params.inertiaWeight);
    printf("Cognitive Weight: %.4f\n", params.cognitiveWeight);
    printf("Social Weight: %.4f\n", params.socialWeight);
    printf("Search Space: [%.2f, %.2f]\n", params.minBound, params.maxBound);
    printf("\n");
    
    // Run PSO on CPU and measure time
    printf("Running PSO on CPU...\n");
    auto cpu_start = std::chrono::high_resolution_clock::now();
    float cpuBestFitness = runPSOonCPU(params);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_duration = cpu_end - cpu_start;
    
    // Run PSO on GPU and measure time
    printf("\nRunning PSO on GPU...\n");
    auto gpu_start = std::chrono::high_resolution_clock::now();
    float gpuBestFitness = runPSOonGPU(params);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> gpu_duration = gpu_end - gpu_start;
    
    // Print results
    printf("\nResults Summary:\n");
    printf("CPU Execution Time: %.2f ms\n", cpu_duration.count());
    printf("GPU Execution Time: %.2f ms\n", gpu_duration.count());
    printf("Speedup: %.2fx\n", cpu_duration.count() / gpu_duration.count());
    printf("CPU Best Fitness: %.10f\n", cpuBestFitness);
    printf("GPU Best Fitness: %.10f\n", gpuBestFitness);
    
    return 0;
}