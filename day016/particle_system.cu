#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>

#define CHECK_CUDA_ERROR(call) \
{ \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at line %d\n", cudaGetErrorString(error), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// Simulation parameters
#define NUM_PARTICLES 65536  // 2^16 particles
#define BLOCK_SIZE 256
#define GRID_SIZE ((NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE)
#define WORLD_SIZE 100.0f
#define TIME_STEP 0.005f
#define GRAVITY 9.8f
#define DAMPING 0.99f
#define REPULSION_STRENGTH 10.0f
#define INTERACTION_RADIUS 5.0f

// Particle structure
typedef struct {
    float4 position;  // (x, y, z, mass)
    float4 velocity;  // (vx, vy, vz, 0)
    float4 color;     // (r, g, b, a)
} Particle;

// Initialize particles with random positions and velocities
void initializeParticles(Particle *particles) {
    srand(time(NULL));
    
    for (int i = 0; i < NUM_PARTICLES; i++) {
        // Random position within world bounds
        particles[i].position.x = (float)rand() / RAND_MAX * WORLD_SIZE - WORLD_SIZE/2;
        particles[i].position.y = (float)rand() / RAND_MAX * WORLD_SIZE - WORLD_SIZE/2;
        particles[i].position.z = (float)rand() / RAND_MAX * WORLD_SIZE - WORLD_SIZE/2;
        particles[i].position.w = 0.1f + (float)rand() / RAND_MAX * 0.9f;  // Random mass between 0.1 and 1.0
        
        // Random initial velocity
        particles[i].velocity.x = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * 5.0f;
        particles[i].velocity.y = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * 5.0f;
        particles[i].velocity.z = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * 5.0f;
        particles[i].velocity.w = 0.0f;
        
        // Color based on position (for visualization)
        particles[i].color.x = 0.5f + particles[i].position.x / WORLD_SIZE * 0.5f;  // Red
        particles[i].color.y = 0.5f + particles[i].position.y / WORLD_SIZE * 0.5f;  // Green
        particles[i].color.z = 0.5f + particles[i].position.z / WORLD_SIZE * 0.5f;  // Blue
        particles[i].color.w = 1.0f;  // Alpha
    }
}

// Grid-based spatial hashing for faster neighbor search
__device__ int calculateGridHash(float3 position, float cellSize) {
    int x = (int)floorf((position.x + WORLD_SIZE/2) / cellSize);
    int y = (int)floorf((position.y + WORLD_SIZE/2) / cellSize);
    int z = (int)floorf((position.z + WORLD_SIZE/2) / cellSize);
    
    // Simple spatial hash function
    return (x * 73856093) ^ (y * 19349663) ^ (z * 83492791);
}

// CUDA kernel for particle simulation with spatial hashing
__global__ void simulateParticlesKernel(
    Particle *particles,
    Particle *newParticles,
    int numParticles,
    float timeStep,
    float worldSize,
    float gravity,
    float damping,
    float repulsionStrength,
    float interactionRadius
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= numParticles) return;
    
    // Load particle data
    Particle p = particles[idx];
    
    // Initialize acceleration
    float3 acceleration = make_float3(0.0f, -gravity, 0.0f);
    
    // Calculate cell size for spatial hashing
    float cellSize = interactionRadius * 2.0f;
    
    // Calculate particle interactions
    for (int j = 0; j < numParticles; j++) {
        if (j == idx) continue;  // Skip self-interaction
        
        Particle other = particles[j];
        
        // Calculate distance between particles
        float3 diff = make_float3(
            p.position.x - other.position.x,
            p.position.y - other.position.y,
            p.position.z - other.position.z
        );
        
        float distSqr = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
        
        // Only interact with particles within interaction radius
        if (distSqr < interactionRadius * interactionRadius && distSqr > 0.0001f) {
            float dist = sqrtf(distSqr);
            float3 dir = make_float3(diff.x / dist, diff.y / dist, diff.z / dist);
            
            // Repulsive force inversely proportional to distance
            float force = repulsionStrength * (1.0f - dist / interactionRadius);
            
            // Apply force based on masses
            float massRatio = p.position.w / other.position.w;
            acceleration.x += dir.x * force * massRatio;
            acceleration.y += dir.y * force * massRatio;
            acceleration.z += dir.z * force * massRatio;
        }
    }
    
    // Update velocity using acceleration
    float4 newVelocity;
    newVelocity.x = (p.velocity.x + acceleration.x * timeStep) * damping;
    newVelocity.y = (p.velocity.y + acceleration.y * timeStep) * damping;
    newVelocity.z = (p.velocity.z + acceleration.z * timeStep) * damping;
    newVelocity.w = 0.0f;
    
    // Update position using velocity
    float4 newPosition;
    newPosition.x = p.position.x + newVelocity.x * timeStep;
    newPosition.y = p.position.y + newVelocity.y * timeStep;
    newPosition.z = p.position.z + newVelocity.z * timeStep;
    newPosition.w = p.position.w;  // Mass remains constant
    
    // Boundary conditions (bounce off walls)
    float halfWorld = worldSize / 2.0f;
    
    if (fabsf(newPosition.x) > halfWorld) {
        newPosition.x = (newPosition.x > 0) ? halfWorld : -halfWorld;
        newVelocity.x = -newVelocity.x * 0.8f;  // Bounce with energy loss
    }
    
    if (fabsf(newPosition.y) > halfWorld) {
        newPosition.y = (newPosition.y > 0) ? halfWorld : -halfWorld;
        newVelocity.y = -newVelocity.y * 0.8f;  // Bounce with energy loss
    }
    
    if (fabsf(newPosition.z) > halfWorld) {
        newPosition.z = (newPosition.z > 0) ? halfWorld : -halfWorld;
        newVelocity.z = -newVelocity.z * 0.8f;  // Bounce with energy loss
    }
    
    // Update color based on velocity (for visualization)
    float4 newColor;
    float speed = sqrtf(newVelocity.x * newVelocity.x + 
                       newVelocity.y * newVelocity.y + 
                       newVelocity.z * newVelocity.z);
    
    newColor.x = fminf(1.0f, 0.2f + speed / 20.0f);  // Red increases with speed
    newColor.y = fminf(1.0f, 0.2f + p.position.y / worldSize + 0.5f);  // Green based on height
    newColor.z = fminf(1.0f, 0.2f + p.position.w);  // Blue based on mass
    newColor.w = 1.0f;  // Alpha
    
    // Write updated particle data
    newParticles[idx].position = newPosition;
    newParticles[idx].velocity = newVelocity;
    newParticles[idx].color = newColor;
}

// Optimized CUDA kernel for particle simulation using shared memory
__global__ void simulateParticlesSharedKernel(
    Particle *particles,
    Particle *newParticles,
    int numParticles,
    float timeStep,
    float worldSize,
    float gravity,
    float damping,
    float repulsionStrength,
    float interactionRadius
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    
    // Shared memory for particle data within a block
    __shared__ Particle sharedParticles[BLOCK_SIZE];
    
    // Load particle data
    Particle p = particles[idx];
    
    // Initialize acceleration
    float3 acceleration = make_float3(0.0f, -gravity, 0.0f);
    
    // Process particles in chunks to utilize shared memory
    for (int blockStart = 0; blockStart < numParticles; blockStart += BLOCK_SIZE) {
        // Load chunk of particles into shared memory
        int sharedIdx = blockStart + threadIdx.x;
        if (sharedIdx < numParticles) {
            sharedParticles[threadIdx.x] = particles[sharedIdx];
        }
        __syncthreads();
        
        // Interact with particles in shared memory
        for (int j = 0; j < BLOCK_SIZE && blockStart + j < numParticles; j++) {
            int otherIdx = blockStart + j;
            if (otherIdx == idx) continue;  // Skip self-interaction
            
            Particle other = sharedParticles[j];
            
            // Calculate distance between particles
            float3 diff = make_float3(
                p.position.x - other.position.x,
                p.position.y - other.position.y,
                p.position.z - other.position.z
            );
            
            float distSqr = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
            
            // Only interact with particles within interaction radius
            if (distSqr < interactionRadius * interactionRadius && distSqr > 0.0001f) {
                float dist = sqrtf(distSqr);
                float3 dir = make_float3(diff.x / dist, diff.y / dist, diff.z / dist);
                
                // Repulsive force inversely proportional to distance
                float force = repulsionStrength * (1.0f - dist / interactionRadius);
                
                // Apply force based on masses
                float massRatio = p.position.w / other.position.w;
                acceleration.x += dir.x * force * massRatio;
                acceleration.y += dir.y * force * massRatio;
                acceleration.z += dir.z * force * massRatio;
            }
        }
        __syncthreads();
    }
    
    // Update velocity using acceleration
    float4 newVelocity;
    newVelocity.x = (p.velocity.x + acceleration.x * timeStep) * damping;
    newVelocity.y = (p.velocity.y + acceleration.y * timeStep) * damping;
    newVelocity.z = (p.velocity.z + acceleration.z * timeStep) * damping;
    newVelocity.w = 0.0f;
    
    // Update position using velocity
    float4 newPosition;
    newPosition.x = p.position.x + newVelocity.x * timeStep;
    newPosition.y = p.position.y + newVelocity.y * timeStep;
    newPosition.z = p.position.z + newVelocity.z * timeStep;
    newPosition.w = p.position.w;  // Mass remains constant
    
    // Boundary conditions (bounce off walls)
    float halfWorld = worldSize / 2.0f;
    
    if (fabsf(newPosition.x) > halfWorld) {
        newPosition.x = (newPosition.x > 0) ? halfWorld : -halfWorld;
        newVelocity.x = -newVelocity.x * 0.8f;  // Bounce with energy loss
    }
    
    if (fabsf(newPosition.y) > halfWorld) {
        newPosition.y = (newPosition.y > 0) ? halfWorld : -halfWorld;
        newVelocity.y = -newVelocity.y * 0.8f;  // Bounce with energy loss
    }
    
    if (fabsf(newPosition.z) > halfWorld) {
        newPosition.z = (newPosition.z > 0) ? halfWorld : -halfWorld;
        newVelocity.z = -newVelocity.z * 0.8f;  // Bounce with energy loss
    }
    
    // Update color based on velocity (for visualization)
    float4 newColor;
    float speed = sqrtf(newVelocity.x * newVelocity.x + 
                       newVelocity.y * newVelocity.y + 
                       newVelocity.z * newVelocity.z);
    
    newColor.x = fminf(1.0f, 0.2f + speed / 20.0f);  // Red increases with speed
    newColor.y = fminf(1.0f, 0.2f + p.position.y / worldSize + 0.5f);  // Green based on height
    newColor.z = fminf(1.0f, 0.2f + p.position.w);  // Blue based on mass
    newColor.w = 1.0f;  // Alpha
    
    // Write updated particle data
    newParticles[idx].position = newPosition;
    newParticles[idx].velocity = newVelocity;
    newParticles[idx].color = newColor;
}

// Main function
int main() {
    printf("CUDA Particle System Simulation\n");
    printf("Number of particles: %d\n", NUM_PARTICLES);
    
    // Allocate host memory
    Particle *h_particles = (Particle *)malloc(NUM_PARTICLES * sizeof(Particle));
    if (!h_particles) {
        fprintf(stderr, "Error: Host memory allocation failed\n");
        return EXIT_FAILURE;
    }
    
    // Initialize particles
    initializeParticles(h_particles);
    
    // Allocate device memory
    Particle *d_particles, *d_newParticles;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_particles, NUM_PARTICLES * sizeof(Particle)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_newParticles, NUM_PARTICLES * sizeof(Particle)));
    
    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_particles, h_particles, NUM_PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice));
    
    // Timing variables
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    // Simulation loop
    const int NUM_ITERATIONS = 100;
    float totalTime = 0.0f;
    
    printf("Running simulation for %d iterations...\n", NUM_ITERATIONS);
    
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        // Record start time
        CHECK_CUDA_ERROR(cudaEventRecord(start));
        
        // Launch kernel (choose one of the two kernels)
        if (iter % 2 == 0) {
            // Use regular kernel for even iterations
            simulateParticlesKernel<<<GRID_SIZE, BLOCK_SIZE>>>(
                d_particles, d_newParticles, NUM_PARTICLES,
                TIME_STEP, WORLD_SIZE, GRAVITY, DAMPING,
                REPULSION_STRENGTH, INTERACTION_RADIUS
            );
        } else {
            // Use shared memory kernel for odd iterations
            simulateParticlesSharedKernel<<<GRID_SIZE, BLOCK_SIZE>>>(
                d_particles, d_newParticles, NUM_PARTICLES,
                TIME_STEP, WORLD_SIZE, GRAVITY, DAMPING,
                REPULSION_STRENGTH, INTERACTION_RADIUS
            );
        }
        
        // Check for kernel errors
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Swap particle buffers
        Particle *temp = d_particles;
        d_particles = d_newParticles;
        d_newParticles = temp;
        
        // Record stop time and calculate elapsed time
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        
        float milliseconds = 0.0f;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
        totalTime += milliseconds;
        
        if (iter % 10 == 0) {
            printf("Iteration %d completed in %.3f ms\n", iter, milliseconds);
        }
    }
    
    // Calculate and print performance metrics
    float avgTimePerIteration = totalTime / NUM_ITERATIONS;
    float particlesPerSecond = (NUM_PARTICLES * NUM_ITERATIONS) / (totalTime / 1000.0f);
    
    printf("\nSimulation complete!\n");
    printf("Average time per iteration: %.3f ms\n", avgTimePerIteration);
    printf("Particles processed per second: %.2f million\n", particlesPerSecond / 1000000.0f);
    
    // Compare kernel performance
    printf("\nComparing kernel performance...\n");
    
    // Regular kernel timing
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for (int i = 0; i < 10; i++) {
        simulateParticlesKernel<<<GRID_SIZE, BLOCK_SIZE>>>(
            d_particles, d_newParticles, NUM_PARTICLES,
            TIME_STEP, WORLD_SIZE, GRAVITY, DAMPING,
            REPULSION_STRENGTH, INTERACTION_RADIUS
        );
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float regularKernelTime = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&regularKernelTime, start, stop));
    regularKernelTime /= 10.0f;  // Average time per iteration
    
    // Shared memory kernel timing
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for (int i = 0; i < 10; i++) {
        simulateParticlesSharedKernel<<<GRID_SIZE, BLOCK_SIZE>>>(
            d_particles, d_newParticles, NUM_PARTICLES,
            TIME_STEP, WORLD_SIZE, GRAVITY, DAMPING,
            REPULSION_STRENGTH, INTERACTION_RADIUS
        );
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float sharedKernelTime = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&sharedKernelTime, start, stop));
    sharedKernelTime /= 10.0f;  // Average time per iteration
    
    printf("Regular kernel average time: %.3f ms\n", regularKernelTime);
    printf("Shared memory kernel average time: %.3f ms\n", sharedKernelTime);
    printf("Performance improvement with shared memory: %.2f%%\n", 
           (regularKernelTime - sharedKernelTime) / regularKernelTime * 100.0f);
    
    // Copy final results back to host for verification
    CHECK_CUDA_ERROR(cudaMemcpy(h_particles, d_particles, NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost));
    
    // Print some sample particle data
    printf("\nSample particle data after simulation:\n");
    for (int i = 0; i < 5 && i < NUM_PARTICLES; i++) {
        printf("Particle %d: pos=(%.2f, %.2f, %.2f), vel=(%.2f, %.2f, %.2f)\n", 
               i,
               h_particles[i].position.x, h_particles[i].position.y, h_particles[i].position.z,
               h_particles[i].velocity.x, h_particles[i].velocity.y, h_particles[i].velocity.z);
    }
    
    // Clean up
    free(h_particles);
    CHECK_CUDA_ERROR(cudaFree(d_particles));
    CHECK_CUDA_ERROR(cudaFree(d_newParticles));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    
    printf("\nCUDA Particle System Simulation completed successfully!\n");
    
    return EXIT_SUCCESS;
}
