cmake_minimum_required(VERSION 3.10)
project(day011_merge_sort CUDA)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA package
find_package(CUDA REQUIRED)

# Add executable
add_executable(merge_sort merge_sort.cu)

# Set CUDA architectures - Jetson Nano has compute capability 5.3
set_target_properties(merge_sort PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "53")

# Install target
install(TARGETS merge_sort DESTINATION bin)
