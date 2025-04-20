#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>

#define CHECK_CUDA_ERROR(call) \
{ \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
}

// Simulation parameters - Kept smaller for faster testing/analysis if needed, but use Day 16 values for comparison
#define NUM_PARTICLES 65536  // 2^16 particles (Same as Day 16 for direct comparison)
#define BLOCK_SIZE 256       // Common block size, balances parallelism and resource usage
#define GRID_SIZE ((NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE)
#define WORLD_SIZE 100.0f
#define TIME_STEP 0.005f
#define GRAVITY 9.8f
#define DAMPING 0.99f
#define REPULSION_STRENGTH 10.0f
#define INTERACTION_RADIUS_SQR (5.0f * 5.0f) // Use squared radius for comparison

// Particle structure (aligning members might be considered for other access patterns, but likely okay here)
typedef struct {
    float4 position;  // (x, y, z, mass) - 16 bytes
    float4 velocity;  // (vx, vy, vz, 0) - 16 bytes
    // Removed color for simplicity in optimization focus, can be added back if visualization needed
} Particle; // Total size: 32 bytes

// Initialize particles with random positions and velocities
void initializeParticles(Particle *particles, int numParticles, float worldSize) {
    srand(time(NULL));
    for (int i = 0; i < numParticles; i++) {
        particles[i].position.x = (float)rand() / RAND_MAX * worldSize - worldSize / 2.0f;
        particles[i].position.y = (float)rand() / RAND_MAX * worldSize - worldSize / 2.0f;
        particles[i].position.z = (float)rand() / RAND_MAX * worldSize - worldSize / 2.0f;
        particles[i].position.w = 0.1f + (float)rand() / RAND_MAX * 0.9f; // Mass

        particles[i].velocity.x = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * 5.0f;
        particles[i].velocity.y = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * 5.0f;
        particles[i].velocity.z = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * 5.0f;
        particles[i].velocity.w = 0.0f;
    }
}

// --- Kernel 1: Global Memory N-Body Interaction ---
// Each thread computes interactions for one particle by reading all other particles from global memory.
__global__ void simulateParticlesGlobalKernel(
    const Particle *particles, // Input particles (read-only)
    Particle *newParticles,    // Output particles
    int numParticles,
    float timeStep,
    float worldSize,
    float gravity,
    float damping,
    float repulsionStrength,
    float interactionRadiusSqr // Use squared radius
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    // Load primary particle data into registers
    float4 p_pos = particles[idx].position;
    float4 p_vel = particles[idx].velocity;
    float p_mass = p_pos.w;

    // Initialize acceleration (gravity)
    float3 acceleration = make_float3(0.0f, -gravity, 0.0f);

    // Calculate interactions with all other particles (N^2 complexity)
    for (int j = 0; j < numParticles; j++) {
        if (j == idx) continue; // Skip self-interaction

        // Read other particle's position from global memory
        float4 other_pos = particles[j].position;
        float other_mass = other_pos.w;

        // Calculate distance squared
        float dx = p_pos.x - other_pos.x;
        float dy = p_pos.y - other_pos.y;
        float dz = p_pos.z - other_pos.z;
        float distSqr = dx * dx + dy * dy + dz * dz;

        // Interact if within radius and not exactly overlapping
        if (distSqr < interactionRadiusSqr && distSqr > 1e-6f) {
            float dist = sqrtf(distSqr);
            float invDist = 1.0f / dist;

            // Normalized direction vector
            float3 dir = make_float3(dx * invDist, dy * invDist, dz * invDist);

            // Repulsive force calculation (simplified model)
            // Force = strength * (1 - dist/radius) -> F = strength * (radius - dist) / radius
            // We use interactionRadius = sqrt(interactionRadiusSqr)
            float forceMagnitude = repulsionStrength * (sqrtf(interactionRadiusSqr) - dist) / sqrtf(interactionRadiusSqr);

            // Apply force: a = F/m. Here we scale force by mass ratio for simplicity, assuming force acts equally.
            // A more physically correct model might differ, but this matches Day 16's logic.
            float massRatio = (other_mass > 1e-6f) ? p_mass / other_mass : 1.0f; // Avoid division by zero
            acceleration.x += dir.x * forceMagnitude * massRatio;
            acceleration.y += dir.y * forceMagnitude * massRatio;
            acceleration.z += dir.z * forceMagnitude * massRatio;
        }
    }

    // Update velocity (Euler integration)
    float4 newVelocity;
    newVelocity.x = (p_vel.x + acceleration.x * timeStep) * damping;
    newVelocity.y = (p_vel.y + acceleration.y * timeStep) * damping;
    newVelocity.z = (p_vel.z + acceleration.z * timeStep) * damping;
    newVelocity.w = 0.0f;

    // Update position (Euler integration)
    float4 newPosition;
    newPosition.x = p_pos.x + newVelocity.x * timeStep;
    newPosition.y = p_pos.y + newVelocity.y * timeStep;
    newPosition.z = p_pos.z + newVelocity.z * timeStep;
    newPosition.w = p_mass; // Mass is constant

    // Boundary conditions (simple bounce)
    float halfWorld = worldSize / 2.0f;
    float bounceDamping = 0.8f;
    if (fabsf(newPosition.x) > halfWorld) {
        newPosition.x = copysignf(halfWorld, newPosition.x);
        newVelocity.x *= -bounceDamping;
    }
    if (fabsf(newPosition.y) > halfWorld) {
        newPosition.y = copysignf(halfWorld, newPosition.y);
        newVelocity.y *= -bounceDamping;
    }
    if (fabsf(newPosition.z) > halfWorld) {
        newPosition.z = copysignf(halfWorld, newPosition.z);
        newVelocity.z *= -bounceDamping;
    }

    // Write updated particle data to global memory
    newParticles[idx].position = newPosition;
    newParticles[idx].velocity = newVelocity;
}


// --- Kernel 2: Shared Memory Optimized N-Body Interaction ---
// Each thread computes interactions for one particle.
// Particles are loaded into shared memory in chunks (tiles) to reduce global memory reads.
__global__ void simulateParticlesSharedKernel(
    const Particle *particles, // Input particles (read-only)
    Particle *newParticles,    // Output particles
    int numParticles,
    float timeStep,
    float worldSize,
    float gravity,
    float damping,
    float repulsionStrength,
    float interactionRadiusSqr // Use squared radius
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    // Load primary particle data into registers (read once from global memory)
    float4 p_pos = particles[idx].position;
    float4 p_vel = particles[idx].velocity;
    float p_mass = p_pos.w;

    // Shared memory tile for holding a chunk of particles
    // Size: BLOCK_SIZE * sizeof(Particle) = 256 * 32 bytes = 8192 bytes (8 KB)
    // This fits comfortably within the typical 48KB usable shared memory per block.
    __shared__ Particle sharedParticles[BLOCK_SIZE];

    // Initialize acceleration (gravity)
    float3 acceleration = make_float3(0.0f, -gravity, 0.0f);

    // Iterate through the particles in tiles (chunks of BLOCK_SIZE)
    for (int tile = 0; tile < gridDim.x; tile++) { // gridDim.x = number of blocks
        // Calculate the global index of the particle this thread should load into shared memory
        int sharedLoadIdx = tile * blockDim.x + threadIdx.x;

        // Load particle data into shared memory (coalesced global read)
        if (sharedLoadIdx < numParticles) {
            sharedParticles[threadIdx.x] = particles[sharedLoadIdx];
        } else {
            // Pad with dummy data if beyond numParticles to avoid uninitialized reads later
            // (Could assign infinite mass or zero position, depending on interaction logic)
             sharedParticles[threadIdx.x].position = make_float4(0.f, 0.f, 0.f, HUGE_VALF); // Indicate invalid/non-interacting
             sharedParticles[threadIdx.x].velocity = make_float4(0.f, 0.f, 0.f, 0.f);
        }

        // Synchronize within the block to ensure all threads have finished loading into shared memory
        __syncthreads();

        // --- Interaction Phase ---
        // Each thread calculates interactions between its primary particle (p_pos, p_vel)
        // and all particles currently residing in the shared memory tile.
        for (int j = 0; j < BLOCK_SIZE; j++) {
            // Index of the 'other' particle within the current tile (global index is tile * BLOCK_SIZE + j)
            int otherGlobalIdx = tile * blockDim.x + j;

            // Avoid self-interaction and interaction with padding particles
             if (otherGlobalIdx == idx || otherGlobalIdx >= numParticles) continue;

            // Read 'other' particle data from shared memory (fast access)
            float4 other_pos = sharedParticles[j].position; // *** READ FROM SHARED MEMORY ***
            float other_mass = other_pos.w;

            // Calculate distance squared
            float dx = p_pos.x - other_pos.x;
            float dy = p_pos.y - other_pos.y;
            float dz = p_pos.z - other_pos.z;
            float distSqr = dx * dx + dy * dy + dz * dz;

            // Interact if within radius and not exactly overlapping
            if (distSqr < interactionRadiusSqr && distSqr > 1e-6f) {
                float dist = sqrtf(distSqr);
                float invDist = 1.0f / dist;
                float3 dir = make_float3(dx * invDist, dy * invDist, dz * invDist);

                float forceMagnitude = repulsionStrength * (sqrtf(interactionRadiusSqr) - dist) / sqrtf(interactionRadiusSqr);

                float massRatio = (other_mass > 1e-6f) ? p_mass / other_mass : 1.0f;
                acceleration.x += dir.x * forceMagnitude * massRatio;
                acceleration.y += dir.y * forceMagnitude * massRatio;
                acceleration.z += dir.z * forceMagnitude * massRatio;
            }
            // --- Bank Conflict Analysis ---
            // The read `sharedParticles[j]` is the critical access.
            // Within a warp (32 threads), all threads access `sharedParticles[j]` for the *same* `j`
            // in the same iteration of the inner loop. This results in a broadcast from the
            // relevant bank(s) holding `sharedParticles[j]`, which is efficient (no bank conflict).
            // The `Particle` struct (32 bytes) spans 32/4 = 8 banks. Accessing `sharedParticles[j]` reads these 8 banks.
            // Since all threads in the warp read the *same* 8 banks for the same `j`, it's okay.
            // If threads accessed `sharedParticles[threadIdx.x + j]` or similar patterns *within the same instruction*,
            // conflicts could arise if `sizeof(Particle)` was a multiple of `32 * 4 = 128` bytes,
            // or if the access stride caused multiple threads to hit the same bank for different addresses.
            // The current `sharedParticles[j]` access pattern is generally considered bank-conflict-free.
        }

        // Synchronize within the block before loading the next tile.
        // Ensures all threads finish calculations using the current tile before it's overwritten.
        __syncthreads();
    }

    // --- Update Phase (same as global kernel) ---
    // Update velocity
    float4 newVelocity;
    newVelocity.x = (p_vel.x + acceleration.x * timeStep) * damping;
    newVelocity.y = (p_vel.y + acceleration.y * timeStep) * damping;
    newVelocity.z = (p_vel.z + acceleration.z * timeStep) * damping;
    newVelocity.w = 0.0f;

    // Update position
    float4 newPosition;
    newPosition.x = p_pos.x + newVelocity.x * timeStep;
    newPosition.y = p_pos.y + newVelocity.y * timeStep;
    newPosition.z = p_pos.z + newVelocity.z * timeStep;
    newPosition.w = p_mass;

    // Boundary conditions
    float halfWorld = worldSize / 2.0f;
    float bounceDamping = 0.8f;
     if (fabsf(newPosition.x) > halfWorld) {
        newPosition.x = copysignf(halfWorld, newPosition.x);
        newVelocity.x *= -bounceDamping;
    }
    if (fabsf(newPosition.y) > halfWorld) {
        newPosition.y = copysignf(halfWorld, newPosition.y);
        newVelocity.y *= -bounceDamping;
    }
    if (fabsf(newPosition.z) > halfWorld) {
        newPosition.z = copysignf(halfWorld, newPosition.z);
        newVelocity.z *= -bounceDamping;
    }

    // Write updated particle data to global memory
    newParticles[idx].position = newPosition;
    newParticles[idx].velocity = newVelocity;
}


// --- Main Function ---
int main(int argc, char **argv) {
    printf("Day 42: N-Body Simulation Optimization (Shared Memory)\n");
    printf("-----------------------------------------------------\n");
    printf("Number of particles: %d\n", NUM_PARTICLES);
    printf("Block size: %d\n", BLOCK_SIZE);
    printf("Grid size: %d\n", GRID_SIZE);

    // Allocate host memory
    Particle *h_particles = (Particle *)malloc(NUM_PARTICLES * sizeof(Particle));
    if (!h_particles) {
        fprintf(stderr, "Error: Host memory allocation failed (h_particles)\n");
        return EXIT_FAILURE;
    }

    // Initialize particles on host
    initializeParticles(h_particles, NUM_PARTICLES, WORLD_SIZE);

    // Allocate device memory (double buffer)
    Particle *d_particles_in, *d_particles_out;
    size_t memSize = NUM_PARTICLES * sizeof(Particle);
    printf("Allocating %.2f MB on device...\n", (float)memSize * 2 / (1024.0f * 1024.0f));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_particles_in, memSize));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_particles_out, memSize));

    // Copy initial data from host to device (input buffer)
    printf("Copying initial data to device...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(d_particles_in, h_particles, memSize, cudaMemcpyHostToDevice));

    // Timing events
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // --- Benchmarking ---
    const int WARMUP_ITERATIONS = 10;
    const int BENCHMARK_ITERATIONS = 100;
    float totalTimeGlobal = 0.0f;
    float totalTimeShared = 0.0f;

    printf("Warming up kernels (%d iterations)...\n", WARMUP_ITERATIONS);
    for (int iter = 0; iter < WARMUP_ITERATIONS; iter++) {
         // Alternate buffers
        Particle *d_in = (iter % 2 == 0) ? d_particles_in : d_particles_out;
        Particle *d_out = (iter % 2 == 0) ? d_particles_out : d_particles_in;

        // Run global kernel
        simulateParticlesGlobalKernel<<<GRID_SIZE, BLOCK_SIZE>>>(
            d_in, d_out, NUM_PARTICLES, TIME_STEP, WORLD_SIZE, GRAVITY, DAMPING,
            REPULSION_STRENGTH, INTERACTION_RADIUS_SQR);
        CHECK_CUDA_ERROR(cudaGetLastError()); // Check for launch errors

         // Run shared kernel
        simulateParticlesSharedKernel<<<GRID_SIZE, BLOCK_SIZE>>>(
            d_in, d_out, NUM_PARTICLES, TIME_STEP, WORLD_SIZE, GRAVITY, DAMPING,
            REPULSION_STRENGTH, INTERACTION_RADIUS_SQR);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }
     CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for warmup to finish
     printf("Warmup complete.\n");


    printf("Benchmarking Global Memory Kernel (%d iterations)...\n", BENCHMARK_ITERATIONS);
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for (int iter = 0; iter < BENCHMARK_ITERATIONS; iter++) {
         Particle *d_in = (iter % 2 == 0) ? d_particles_in : d_particles_out;
         Particle *d_out = (iter % 2 == 0) ? d_particles_out : d_particles_in;
        simulateParticlesGlobalKernel<<<GRID_SIZE, BLOCK_SIZE>>>(
            d_in, d_out, NUM_PARTICLES, TIME_STEP, WORLD_SIZE, GRAVITY, DAMPING,
            REPULSION_STRENGTH, INTERACTION_RADIUS_SQR);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&totalTimeGlobal, start, stop));
    printf("Global Kernel Average Time: %.3f ms\n", totalTimeGlobal / BENCHMARK_ITERATIONS);


    // Need to copy result back to input buffer before running shared kernel benchmark if we want identical start state
    // However, for pure timing, using the last state is acceptable. Let's keep it simple for now.
    // If results verification were needed, we'd copy d_particles_out back to d_particles_in here.


    printf("Benchmarking Shared Memory Kernel (%d iterations)...\n", BENCHMARK_ITERATIONS);
    CHECK_CUDA_ERROR(cudaEventRecord(start));
     for (int iter = 0; iter < BENCHMARK_ITERATIONS; iter++) {
         Particle *d_in = (iter % 2 == 0) ? d_particles_in : d_particles_out;
         Particle *d_out = (iter % 2 == 0) ? d_particles_out : d_particles_in;
        simulateParticlesSharedKernel<<<GRID_SIZE, BLOCK_SIZE>>>(
             d_in, d_out, NUM_PARTICLES, TIME_STEP, WORLD_SIZE, GRAVITY, DAMPING,
            REPULSION_STRENGTH, INTERACTION_RADIUS_SQR);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&totalTimeShared, start, stop));
    printf("Shared Kernel Average Time: %.3f ms\n", totalTimeShared / BENCHMARK_ITERATIONS);


    // --- Performance Comparison ---
    printf("\n--- Performance Results ---\n");
    float avgTimeGlobal = totalTimeGlobal / BENCHMARK_ITERATIONS;
    float avgTimeShared = totalTimeShared / BENCHMARK_ITERATIONS;
    printf("Avg. Global Kernel Time: %.3f ms\n", avgTimeGlobal);
    printf("Avg. Shared Kernel Time: %.3f ms\n", avgTimeShared);
    if (avgTimeGlobal > 0.0f) {
        float speedup = avgTimeGlobal / avgTimeShared;
        float improvement = (avgTimeGlobal - avgTimeShared) / avgTimeGlobal * 100.0f;
        printf("Speedup (Shared vs Global): %.2fx\n", speedup);
        printf("Performance Improvement: %.2f%%\n", improvement);
    } else {
        printf("Could not calculate speedup (Global kernel time was zero).\n");
    }

    // Copy final results back to host (optional, for verification)
    // Let's copy from the last output buffer used by the shared kernel benchmark
    // Particle *d_final_out = (BENCHMARK_ITERATIONS % 2 == 0) ? d_particles_in : d_particles_out; // This depends on which kernel ran last! Let's assume shared kernel ran last.
    Particle *d_final_out = (BENCHMARK_ITERATIONS % 2 != 0) ? d_particles_in : d_particles_out; // If benchmark_iterations is odd, last write was to d_particles_in
    
    printf("\nCopying final data back to host...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(h_particles, d_final_out, memSize, cudaMemcpyDeviceToHost));


    // Print some sample particle data
    printf("\nSample particle data after simulation:\n");
    for (int i = 0; i < 5 && i < NUM_PARTICLES; i++) {
        printf("Particle %d: pos=(%.2f, %.2f, %.2f), vel=(%.2f, %.2f, %.2f)\n",
               i,
               h_particles[i].position.x, h_particles[i].position.y, h_particles[i].position.z,
               h_particles[i].velocity.x, h_particles[i].velocity.y, h_particles[i].velocity.z);
    }

    // Clean up
    printf("\nCleaning up...\n");
    free(h_particles);
    CHECK_CUDA_ERROR(cudaFree(d_particles_in));
    CHECK_CUDA_ERROR(cudaFree(d_particles_out));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    printf("\nDay 42 Simulation completed successfully!\n");
    return EXIT_SUCCESS;
}
