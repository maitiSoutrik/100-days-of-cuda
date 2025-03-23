cmake_minimum_required(VERSION 3.10)

# Set the project name
project(day013 LANGUAGES CXX CUDA)

# Find CUDA package
find_package(CUDA REQUIRED)

# Set CUDA flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3)

# Add executable
cuda_add_executable(bfs_optimized bfs_optimized.cu)

# Set C++ standard
set_property(TARGET bfs_optimized PROPERTY CXX_STANDARD 14)

# Install target
install(TARGETS bfs_optimized DESTINATION bin)
