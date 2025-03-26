cmake_minimum_required(VERSION 3.10)

# Find CUDA package
find_package(CUDA REQUIRED)

# Set CUDA flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3)

# Add executable
cuda_add_executable(particle_system particle_system.cu)
