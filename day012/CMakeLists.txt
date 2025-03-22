cmake_minimum_required(VERSION 3.10)
project(day012_bfs CUDA C)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA package
find_package(CUDA REQUIRED)

# Enable C99 standard for C files
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Instead of creating a separate library, compile all source files together
add_executable(bfs main.cu bfs_gpu.cu bfs_cpu.c)

# Set CUDA architectures - Jetson Nano has compute capability 5.3
set_target_properties(bfs PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "53")

# Install target
install(TARGETS bfs DESTINATION bin)
