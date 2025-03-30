# Day 21: Particle Swarm Optimization (PSO) with CUDA

## Overview

Today's implementation focuses on Particle Swarm Optimization (PSO), a population-based stochastic optimization technique inspired by social behavior of bird flocking or fish schooling. PSO is widely used for solving complex optimization problems in various domains including machine learning, engineering design, and finance.

## What is Particle Swarm Optimization?

PSO is a computational method that optimizes a problem by iteratively trying to improve a candidate solution with regard to a given measure of quality. It solves a problem by having a population of candidate solutions (particles) and moving these particles around in the search-space according to simple mathematical formulae. The movements of the particles are guided by their own best known position in the search-space as well as the entire swarm's best known position.

## Implementation Details

This implementation includes:

1. **CUDA-accelerated PSO algorithm**: Each particle's position and velocity updates are computed in parallel on the GPU
2. **Test functions**: Several benchmark optimization functions (Sphere, Rastrigin, Rosenbrock) to evaluate the PSO performance
3. **Performance comparison**: CPU vs. GPU implementation with timing measurements
4. **Visualization**: Simple text-based visualization of the optimization progress

## Key CUDA Features Used

- **Global memory**: For storing particle positions, velocities, and fitness values
- **Random number generation**: Using cuRAND for generating random numbers on the GPU
- **Parallel reduction**: For finding the global best position efficiently
- **Atomic operations**: For updating shared best positions without race conditions

## Optimization Functions

The implementation includes several standard benchmark functions:

1. **Sphere function**: A simple unimodal function with a global minimum at the origin
2. **Rastrigin function**: A non-convex function with many local minima, making it challenging for optimization algorithms
3. **Rosenbrock function**: A non-convex function with a narrow valley from the local optimum to the global optimum

## Usage

```bash
# Run PSO on the Sphere function
./pso_optimization --function sphere --dimensions 10 --particles 1024 --iterations 1000

# Run PSO on the Rastrigin function with custom bounds
./pso_optimization --function rastrigin --dimensions 20 --particles 2048 --iterations 2000 --min -5.12 --max 5.12

# Run PSO on the Rosenbrock function
./pso_optimization --function rosenbrock --dimensions 5 --particles 512 --iterations 1500
```

## Results

The implementation demonstrates the effectiveness of GPU acceleration for PSO, showing significant speedup compared to the CPU implementation, especially for large swarm sizes and high-dimensional problems.

### Execution Logs on Jetson Nano

Below are the actual execution results from running the PSO implementation on a Jetson Nano with different test functions and parameters:

#### Sphere Function

```
./day021/pso_optimization --function sphere --dimensions 10 --particles 1024 --iterations 1000
                            
Particle Swarm Optimization (PSO)
=================================
Function: Sphere
Dimensions: 10
Particles: 1024
Iterations: 1000
Inertia Weight: 0.7290
Cognitive Weight: 1.4944
Social Weight: 1.4944
Search Space: [-100.00, 100.00]

Running PSO on CPU...
CPU - Iteration 0: Best fitness = 1435.5155029297
CPU - Iteration 100: Best fitness = 0.0000441346
CPU - Iteration 200: Best fitness = 0.0000000000
CPU - Iteration 300: Best fitness = 0.0000000000
CPU - Iteration 400: Best fitness = 0.0000000000
CPU - Iteration 500: Best fitness = 0.0000000000
CPU - Iteration 600: Best fitness = 0.0000000000
CPU - Iteration 700: Best fitness = 0.0000000000
CPU - Iteration 800: Best fitness = 0.0000000000
CPU - Iteration 900: Best fitness = 0.0000000000
CPU - Iteration 999: Best fitness = 0.0000000000

Running PSO on GPU...
GPU - Iteration 0: Best fitness = 6810.2050781250
GPU - Iteration 100: Best fitness = 0.5079932809
GPU - Iteration 200: Best fitness = 0.0002827399
GPU - Iteration 300: Best fitness = 0.0000003086
GPU - Iteration 400: Best fitness = 0.0000000000
GPU - Iteration 500: Best fitness = 0.0000000000
GPU - Iteration 600: Best fitness = 0.0000000000
GPU - Iteration 700: Best fitness = 0.0000000000
GPU - Iteration 800: Best fitness = 0.0000000000
GPU - Iteration 900: Best fitness = 0.0000000000
GPU - Iteration 999: Best fitness = 0.0000000000

Best solution found:
Fitness: 0.0000000000
Position: [0.000000, 0.000000, -0.000000, 0.000000, 0.000000, 0.000000, -0.000000, -0.000000, -0.000000, -0.000000]

Results Summary:
CPU Execution Time: 2263.29 ms
GPU Execution Time: 1778.10 ms
Speedup: 1.27x
CPU Best Fitness: 0.0000000000
GPU Best Fitness: 0.0000000000
```

#### Rastrigin Function

```
./day021/pso_optimization --function rastrigin --dimensions 20 --particles 2048 --iterations 2000 --min -5.12 --max 5.12
Particle Swarm Optimization (PSO)
=================================
Function: Rastrigin
Dimensions: 20
Particles: 2048
Iterations: 2000
Inertia Weight: 0.7290
Cognitive Weight: 1.4944
Social Weight: 1.4944
Search Space: [-5.12, 5.12]

Running PSO on CPU...
CPU - Iteration 0: Best fitness = 143.1247711182
CPU - Iteration 100: Best fitness = 12.8155031204
CPU - Iteration 200: Best fitness = 6.9755163193
CPU - Iteration 300: Best fitness = 5.9697513580
CPU - Iteration 400: Best fitness = 5.9697494507
CPU - Iteration 500: Best fitness = 4.9748134613
CPU - Iteration 600: Best fitness = 4.9747924805
CPU - Iteration 700: Best fitness = 3.9798374176
CPU - Iteration 800: Best fitness = 3.9798355103
CPU - Iteration 900: Best fitness = 2.9848785400
CPU - Iteration 1000: Best fitness = 2.9848785400
CPU - Iteration 1100: Best fitness = 2.9848785400
CPU - Iteration 1200: Best fitness = 2.9848785400
CPU - Iteration 1300: Best fitness = 2.9848785400
CPU - Iteration 1400: Best fitness = 2.9848785400
CPU - Iteration 1500: Best fitness = 2.9848785400
CPU - Iteration 1600: Best fitness = 2.9848785400
CPU - Iteration 1700: Best fitness = 2.9848785400
CPU - Iteration 1800: Best fitness = 2.9848785400
CPU - Iteration 1900: Best fitness = 2.9848785400
CPU - Iteration 1999: Best fitness = 2.9848785400

Running PSO on GPU...
GPU - Iteration 0: Best fitness = 204.5845794678
GPU - Iteration 100: Best fitness = 76.3169860840
GPU - Iteration 200: Best fitness = 42.6130828857
GPU - Iteration 300: Best fitness = 27.8884277344
GPU - Iteration 400: Best fitness = 20.5232772827
GPU - Iteration 500: Best fitness = 17.5727462769
GPU - Iteration 600: Best fitness = 12.9775724411
GPU - Iteration 700: Best fitness = 11.9925928116
GPU - Iteration 800: Best fitness = 10.9833278656
GPU - Iteration 900: Best fitness = 9.9739923477
GPU - Iteration 1000: Best fitness = 9.9509677887
GPU - Iteration 1100: Best fitness = 9.9495935440
GPU - Iteration 1200: Best fitness = 5.9731941223
GPU - Iteration 1300: Best fitness = 5.9697980881
GPU - Iteration 1400: Best fitness = 5.9697494507
GPU - Iteration 1500: Best fitness = 5.9697494507
GPU - Iteration 1600: Best fitness = 5.9697494507
GPU - Iteration 1700: Best fitness = 5.9697494507
GPU - Iteration 1800: Best fitness = 5.9697494507
GPU - Iteration 1900: Best fitness = 5.9697494507
GPU - Iteration 1999: Best fitness = 5.9697494507

Best solution found:
Fitness: 5.9697494507
Position: [1.990045, 0.000081, 0.000223, -0.994867, -0.000001, -0.000165, -0.000071, 0.000131, -0.000080, -0.994983, 0.000098, -0.000085, -0.000118, 0.000032, 0.000132, -0.000053, 0.000068, 0.000023, -0.000002, 0.000012]

Results Summary:
CPU Execution Time: 21828.54 ms
GPU Execution Time: 4039.40 ms
Speedup: 5.40x
CPU Best Fitness: 2.9848785400
GPU Best Fitness: 5.9697494507
```

#### Rosenbrock Function

```
./day021/pso_optimization --function rosenbrock --dimensions 5 --particles 512 --iterations 1500
Particle Swarm Optimization (PSO)
=================================
Function: Rosenbrock
Dimensions: 5
Particles: 512
Iterations: 1500
Inertia Weight: 0.7290
Cognitive Weight: 1.4944
Social Weight: 1.4944
Search Space: [-100.00, 100.00]

Running PSO on CPU...
CPU - Iteration 0: Best fitness = 737828.8125000000
CPU - Iteration 100: Best fitness = 0.5122001171
CPU - Iteration 200: Best fitness = 0.0641863570
CPU - Iteration 300: Best fitness = 0.0181239024
CPU - Iteration 400: Best fitness = 0.0078318240
CPU - Iteration 500: Best fitness = 0.0028993092
CPU - Iteration 600: Best fitness = 0.0018120720
CPU - Iteration 700: Best fitness = 0.0009622378
CPU - Iteration 800: Best fitness = 0.0006092463
CPU - Iteration 900: Best fitness = 0.0003568319
CPU - Iteration 1000: Best fitness = 0.0002071664
CPU - Iteration 1100: Best fitness = 0.0001469571
CPU - Iteration 1200: Best fitness = 0.0000838594
CPU - Iteration 1300: Best fitness = 0.0000398471
CPU - Iteration 1400: Best fitness = 0.0000224235
CPU - Iteration 1499: Best fitness = 0.0000145945

Running PSO on GPU...
GPU - Iteration 0: Best fitness = 30969664.0000000000
GPU - Iteration 100: Best fitness = 98.7542648315
GPU - Iteration 200: Best fitness = 82.5384750366
GPU - Iteration 300: Best fitness = 72.4692611694
GPU - Iteration 400: Best fitness = 54.3982658386
GPU - Iteration 500: Best fitness = 45.9319458008
GPU - Iteration 600: Best fitness = 41.4702072144
GPU - Iteration 700: Best fitness = 36.6124420166
GPU - Iteration 800: Best fitness = 22.5108718872
GPU - Iteration 900: Best fitness = 21.0278224945
GPU - Iteration 1000: Best fitness = 19.6245307922
GPU - Iteration 1100: Best fitness = 18.4033832550
GPU - Iteration 1200: Best fitness = 14.5460586548
GPU - Iteration 1300: Best fitness = 13.5259885788
GPU - Iteration 1400: Best fitness = 12.6062297821
GPU - Iteration 1499: Best fitness = 11.0640993118

Best solution found:
Fitness: 11.0640993118
Position: [1.196882, 1.438026, 2.028376, 4.061949, 16.518042]

Results Summary:
CPU Execution Time: 900.79 ms
GPU Execution Time: 1414.49 ms
Speedup: 0.64x
CPU Best Fitness: 0.0000145945
GPU Best Fitness: 11.0640993118
```

### Analysis of Results

1. **Sphere Function**: Both CPU and GPU implementations converge to the global minimum (0.0) with the GPU showing a 1.27x speedup. The GPU takes slightly longer to converge initially but achieves the same final result.

2. **Rastrigin Function**: This is a more challenging multimodal function. The GPU implementation shows a significant 5.40x speedup with 2048 particles and 20 dimensions, demonstrating the advantage of GPU parallelism for larger problem sizes. The CPU found a slightly better solution (2.98 vs 5.97), likely due to different random initialization.

3. **Rosenbrock Function**: Interestingly, for this function, the CPU outperforms the GPU (0.64x speedup, meaning the GPU is slower) and finds a much better solution. This could be due to the sequential nature of the Rosenbrock valley, which might benefit less from parallelization with smaller particle counts.

## References

1. Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. Proceedings of ICNN'95 - International Conference on Neural Networks.
2. Clerc, M., & Kennedy, J. (2002). The particle swarm - explosion, stability, and convergence in a multidimensional complex space. IEEE Transactions on Evolutionary Computation.
3. Shi, Y., & Eberhart, R. (1998). A modified particle swarm optimizer. IEEE International Conference on Evolutionary Computation Proceedings.