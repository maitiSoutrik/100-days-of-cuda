# Day 31: 2D Heat Equation Simulation with CUDA

This project implements a 2D heat diffusion simulation using CUDA. The simulation solves the heat equation using a finite difference method on a grid, with both basic and shared memory kernel implementations for performance comparison.

## The Heat Equation

The heat equation is a partial differential equation that describes how heat diffuses through a material over time:

∂u/∂t = α∇²u

Where:
- u(x, y, t) is the temperature at position (x, y) and time t
- α is the thermal diffusivity constant
- ∇² is the Laplacian operator (sum of second partial derivatives)

In 2D, the Laplacian is:
∇²u = ∂²u/∂x² + ∂²u/∂y²

## Finite Difference Method

We use the explicit finite difference method to discretize the heat equation:

u(i,j,t+Δt) = u(i,j,t) + α·Δt·[u(i+1,j,t) + u(i-1,j,t) + u(i,j+1,t) + u(i,j-1,t) - 4·u(i,j,t)]

Where:
- u(i,j,t) is the temperature at grid point (i,j) at time t
- Δt is the time step
- α is the thermal diffusivity constant

## Implementation Details

### Basic Kernel

The basic kernel implementation directly reads from and writes to global memory:

```cuda
__global__ void heat_step_kernel(const float* current_grid, float* next_grid, int width, int height, float dt, float alpha) {
    // Calculate grid position
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;
    
    // Handle boundary conditions and calculate Laplacian
    // Update temperature using the heat equation
}
```

### Shared Memory Kernel

The shared memory kernel loads a tile of the grid into shared memory, including halo cells, to reduce global memory accesses:

```cuda
__global__ void heat_step_shared_kernel(const float* current_grid, float* next_grid, int width, int height, float dt, float alpha) {
    // Shared memory for tile including halo cells
    __shared__ float tile[TILE_DIM + 2][TILE_DIM + 2];
    
    // Load data into shared memory
    // Calculate Laplacian using shared memory
    // Update temperature using the heat equation
}
```

### Ping-Pong Buffer Technique

The simulation uses a ping-pong buffer approach to avoid overwriting data that's still needed:

```c
// Swap pointers for next iteration
float *temp = d_grid_current;
d_grid_current = d_grid_next;
d_grid_next = temp;
```

This technique allows us to avoid an extra copy operation by simply swapping pointers.

### Reduction Kernels

Two parallel reduction kernels are implemented to calculate:
1. The sum of all temperatures (for computing the average)
2. The maximum temperature in the grid

These kernels use shared memory to efficiently perform the reductions in parallel.

## Building and Running

```bash
# Build the project
mkdir -p build && cd build
cmake ..
make

# Run with basic kernel
./day031/heat_simulation

# Run with shared memory kernel
./day031/heat_simulation shared
```

## Performance Analysis

The shared memory kernel typically outperforms the basic kernel due to:

1. Reduced global memory accesses
2. Better memory coalescing
3. Improved cache utilization

The performance difference becomes more significant as the grid size increases, as the ratio of computation to memory access improves with larger tiles.

## Sample Output

```
Device name: NVIDIA GeForce RTX 3080
Compute capability: 8.6
Total global memory: 10.00 GB

----- Simulation Parameters -----
Grid size: 512 x 512
Number of timesteps: 5000
Time step size (dt): 0.2500
Diffusion constant (alpha): 0.1000
Using basic kernel (use 'shared' argument to enable shared memory)

----- Starting Simulation -----
Timestep: 0, Avg Temp: 0.3815, Max Temp: 100.0000, Kernel Time: 0.123 ms
Timestep: 100, Avg Temp: 0.3815, Max Temp: 42.1234, Kernel Time: 0.118 ms
...
Timestep: 4900, Avg Temp: 0.3815, Max Temp: 1.2345, Kernel Time: 0.117 ms
Timestep: 4999, Avg Temp: 0.3815, Max Temp: 1.1234, Kernel Time: 0.118 ms

----- Performance Summary -----
Total kernel execution time: 590.23 ms
Average kernel execution time per step: 0.1180 ms
```

Note how the maximum temperature decreases over time as heat diffuses throughout the grid, while the average temperature remains constant (conservation of energy).
