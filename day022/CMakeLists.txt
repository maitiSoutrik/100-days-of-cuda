cmake_minimum_required(VERSION 3.10)
project(q_learning CUDA CXX)

# Set CUDA architecture based on Jetson Nano
set(CMAKE_CUDA_ARCHITECTURES 53)  # Compute capability 5.3 for Jetson Nano

# Find required packages
find_package(CUDA REQUIRED)

# Set CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++14")

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})

# Add executables
add_executable(q_learning
    q_learning.cu
)

# Link libraries
target_link_libraries(q_learning
    ${CUDA_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    m
)

# Set C++ standard
set_target_properties(q_learning PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
)