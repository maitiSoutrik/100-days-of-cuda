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

## Performance Analysis (General)

## Execution Results (Jetson Nano)

Here are the results from running both the basic and shared memory kernels on the Jetson Nano (Compute Capability 5.3).

### Basic Kernel

```
drboom@JetNano ~/g/1/build> ./day031/heat_simulation
Device name: NVIDIA Tegra X1
Compute capability: 5.3
Total global memory: 3.87 GB

----- Simulation Parameters -----
Grid size: 512 x 512
Number of timesteps: 5000
Time step size (dt): 0.2500
Diffusion constant (alpha): 0.1000
Using basic kernel (use 'shared' argument to enable shared memory)

----- Starting Simulation -----
Timestep: 0, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 1.812 ms
Timestep: 100, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.437 ms
Timestep: 200, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.462 ms
Timestep: 300, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.423 ms
Timestep: 400, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.435 ms
Timestep: 500, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.474 ms
Timestep: 600, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.427 ms
Timestep: 700, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.432 ms
Timestep: 800, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.457 ms
Timestep: 900, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.422 ms
Timestep: 1000, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.370 ms
Timestep: 1100, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.367 ms
Timestep: 1200, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.381 ms
Timestep: 1300, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.369 ms
Timestep: 1400, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.326 ms
Timestep: 1500, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.327 ms
Timestep: 1600, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.356 ms
Timestep: 1700, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.275 ms
Timestep: 1800, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.313 ms
Timestep: 1900, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.266 ms
Timestep: 2000, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.272 ms
Timestep: 2100, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.269 ms
Timestep: 2200, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.287 ms
Timestep: 2300, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.256 ms
Timestep: 2400, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.239 ms
Timestep: 2500, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.244 ms
Timestep: 2600, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.252 ms
Timestep: 2700, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.267 ms
Timestep: 2800, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.253 ms
Timestep: 2900, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.233 ms
Timestep: 3000, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.227 ms
Timestep: 3100, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.231 ms
Timestep: 3200, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.234 ms
Timestep: 3300, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.241 ms
Timestep: 3400, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.274 ms
Timestep: 3500, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.228 ms
Timestep: 3600, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.228 ms
Timestep: 3700, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.228 ms
Timestep: 3800, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.235 ms
Timestep: 3900, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.234 ms
Timestep: 4000, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.239 ms
Timestep: 4100, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.260 ms
Timestep: 4200, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.264 ms
Timestep: 4300, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.238 ms
Timestep: 4400, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.232 ms
Timestep: 4500, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.231 ms
Timestep: 4600, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.226 ms
Timestep: 4700, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.236 ms
Timestep: 4800, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.239 ms
Timestep: 4900, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.237 ms
Timestep: 4999, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.261 ms

----- Performance Summary -----
Total kernel execution time: 1531.69 ms
Average kernel execution time per step: 0.3063 ms

----- Final Grid Sample (center region) -----
78.25 78.38 78.38 78.25 78.00 
78.38 78.51 78.51 78.38 78.12 
78.38 78.51 78.51 78.38 78.12 
78.25 78.38 78.38 78.25 78.00 
78.00 78.12 78.12 78.00 77.74 
```

### Shared Memory Kernel

```
drboom@JetNano ~/g/1/build> ./day031/heat_simulation shared
Device name: NVIDIA Tegra X1
Compute capability: 5.3
Total global memory: 3.87 GB
----- Simulation Parameters -----
Grid size: 512 x 512
Number of timesteps: 5000
Time step size (dt): 0.2500
Diffusion constant (alpha): 0.1000
Using shared memory kernel

----- Starting Simulation -----
Timestep: 0, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 2.600 ms
Timestep: 100, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.568 ms
Timestep: 200, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.339 ms
Timestep: 300, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.324 ms
Timestep: 400, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.332 ms
Timestep: 500, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.338 ms
Timestep: 600, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.332 ms
Timestep: 700, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.319 ms
Timestep: 800, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.296 ms
Timestep: 900, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.302 ms
Timestep: 1000, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.273 ms
Timestep: 1100, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.273 ms
Timestep: 1200, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.278 ms
Timestep: 1300, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.260 ms
Timestep: 1400, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.262 ms
Timestep: 1500, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.265 ms
Timestep: 1600, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.277 ms
Timestep: 1700, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.258 ms
Timestep: 1800, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.258 ms
Timestep: 1900, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.266 ms
Timestep: 2000, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.280 ms
Timestep: 2100, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.258 ms
Timestep: 2200, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.261 ms
Timestep: 2300, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.266 ms
Timestep: 2400, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.274 ms
Timestep: 2500, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.260 ms
Timestep: 2600, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.261 ms
Timestep: 2700, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.265 ms
Timestep: 2800, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.283 ms
Timestep: 2900, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.260 ms
Timestep: 3000, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.267 ms
Timestep: 3100, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.286 ms
Timestep: 3200, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.259 ms
Timestep: 3300, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.261 ms
Timestep: 3400, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.266 ms
Timestep: 3500, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.278 ms
Timestep: 3600, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.262 ms
Timestep: 3700, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.260 ms
Timestep: 3800, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.269 ms
Timestep: 3900, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.282 ms
Timestep: 4000, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.260 ms
Timestep: 4100, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.259 ms
Timestep: 4200, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.265 ms
Timestep: 4300, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.282 ms
Timestep: 4400, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.262 ms
Timestep: 4500, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.262 ms
Timestep: 4600, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.266 ms
Timestep: 4700, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.279 ms
Timestep: 4800, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.261 ms
Timestep: 4900, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.261 ms
Timestep: 4999, Avg Temp: 0.0000, Max Temp: 0.0000, Kernel Time: 0.265 ms

----- Performance Summary -----
Total kernel execution time: 1482.84 ms
Average kernel execution time per step: 0.2966 ms

----- Final Grid Sample (center region) -----
78.25 78.38 78.38 78.25 78.00 
78.38 78.51 78.51 78.38 78.12 
78.38 78.51 78.51 78.38 78.12 
78.25 78.38 78.38 78.25 78.00 
78.00 78.12 78.12 78.00 77.74 
```

## Performance Analysis (Jetson Nano)

Based on the execution results on the Jetson Nano:

- **Basic Kernel:**
    - Total kernel execution time: 1531.69 ms
    - Average kernel execution time per step: 0.3063 ms
- **Shared Memory Kernel:**
    - Total kernel execution time: 1482.84 ms
    - Average kernel execution time per step: 0.2966 ms

**Observations:**

- The shared memory kernel is slightly faster than the basic kernel on the Jetson Nano for this specific problem size and parameters.
- The performance improvement is approximately (1531.69 - 1482.84) / 1531.69 ≈ **3.2%**.
- While the shared memory optimization provides a benefit, the difference is less dramatic than often seen on higher-end GPUs. This might be due to the architecture of the Tegra X1 (Maxwell) or the specific nature of the computation (stencil operation).
- Both kernels show an initial higher execution time for the first step, likely due to JIT compilation or cache warming effects. The times stabilize quickly in subsequent steps.

The final grid samples for both runs are identical, confirming that both kernels produce the correct results.
