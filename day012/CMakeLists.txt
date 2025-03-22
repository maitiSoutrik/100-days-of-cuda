cmake_minimum_required(VERSION 3.10)
project(day012_bfs CUDA)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA package
find_package(CUDA REQUIRED)

# Add executable
add_executable(bfs main.cu bfs_gpu.cu bfs_cpu.c)

# Set CUDA architectures - Jetson Nano has compute capability 5.3
set_target_properties(bfs PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "53")

# Install target
install(TARGETS bfs DESTINATION bin)
