# Day 1: Getting Started with CUDA

## Today's Progress

Today I set up my development environment and wrote my first CUDA kernel to add two vectors. I successfully compiled and ran the code on my Jetson Nano, which was a great first step into CUDA programming.

## Thoughts

Starting with CUDA programming was both exciting and challenging. Setting up the environment took some time, but once everything was in place, writing the vector addition kernel was straightforward. I learned about the basic CUDA programming model, including how to allocate memory on the GPU, transfer data between host and device, launch kernels, and synchronize operations.

The vector addition example is a perfect starting point as it demonstrates the fundamental concepts of parallel programming on the GPU. I was impressed by how the CUDA runtime automatically manages thread blocks and grids, making it relatively easy to parallelize operations.

## Resources Used

- PMPP (Programming Massively Parallel Processors) Book Chapter 3
- NVIDIA CUDA C++ Programming Guide
- NVIDIA CUDA Samples

## Output from Jetson Nano

```bash
drboom@JetNano ~/g/1/day001> ./vector_add 
Device name: NVIDIA Tegra X1
Compute capability: 5.3
Total global memory: 3.87 GB
Vector size: 50000 elements
CUDA kernel launch with 196 blocks of 256 threads
Test PASSED
Sample results:
0.840188 + 0.394383 = 1.234571
0.783099 + 0.798440 = 1.581539
0.911647 + 0.197551 = 1.109199
0.335223 + 0.768230 = 1.103452
0.277775 + 0.553970 = 0.831745
0.477397 + 0.628871 = 1.106268
0.364784 + 0.513401 = 0.878185
0.952230 + 0.916195 = 1.868425
0.635712 + 0.717297 = 1.353009
0.141603 + 0.606969 = 0.748571
Done
```

## Next Steps

Next, I plan to explore more complex CUDA kernels and learn about memory coalescing and shared memory to optimize performance. I also want to implement a matrix multiplication example to understand how to work with multi-dimensional data in CUDA.
