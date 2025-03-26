# Day 16: CUDA Particle System Simulation

## Overview

This project implements a parallel particle system simulation using CUDA. The simulation models thousands of particles interacting with each other through repulsive forces while being affected by gravity and boundary conditions.

## Features

- Simulates 65,536 particles (2^16) in parallel
- Implements particle-particle interactions with repulsive forces
- Applies gravity and boundary conditions (particles bounce off walls)
- Uses two different kernel implementations for performance comparison:
  - Basic global memory implementation
  - Optimized shared memory implementation
- Measures and compares performance between implementations
- Dynamic particle coloring based on velocity, position, and mass

## Implementation Details

### Particle Structure

Each particle has:

- Position (x, y, z) and mass
- Velocity (vx, vy, vz)
- Color (r, g, b, a) for visualization

### Physics Simulation

The simulation includes:

- Gravitational force
- Particle-particle repulsive interactions
- Boundary collision detection and response
- Velocity damping to simulate energy loss

### Optimization Techniques

1. **Shared Memory Usage**: The optimized kernel loads particles into shared memory to reduce global memory access latency
2. **Spatial Hashing**: A grid-based spatial hashing function is implemented to optimize neighbor searches
3. **Memory Coalescing**: Particle data is structured to promote coalesced memory access patterns
4. **Double Buffering**: Two particle buffers are used to avoid race conditions during updates

## Performance Analysis

The code measures and compares performance between the two kernel implementations:

- Regular global memory kernel
- Shared memory optimized kernel

The performance improvement with shared memory is calculated and reported at the end of the simulation.

## Building and Running

```bash
# Navigate to the day016 directory
cd day016

# Build the project
cmake .
make

# Run the simulation
./particle_system
```

## Expected Output

The program will output:

- Simulation parameters
- Performance metrics for each iteration
- Comparison between the two kernel implementations
- Sample particle data after the simulation

## Deployment to Jetson Nano

This project can be deployed to the Jetson Nano using the GitHub Actions workflow. The workflow will automatically build and deploy the code when changes are pushed to the repository.
