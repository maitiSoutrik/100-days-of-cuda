cmake_minimum_required(VERSION 3.10)
project(day032 LANGUAGES CXX CUDA)

# Set CUDA architecture specifically for Jetson Nano
set(CMAKE_CUDA_ARCHITECTURES 53)

# Set optimization level
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Add executable
# Using matrix multiply as the base for demonstrating stream overlap
add_executable(stream_overlap stream_overlap.cu)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Link CUDA runtime library
target_link_libraries(stream_overlap PRIVATE ${CUDA_LIBRARIES})
