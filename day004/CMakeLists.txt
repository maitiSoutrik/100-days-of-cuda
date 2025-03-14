cmake_minimum_required(VERSION 3.10)
project(day004 LANGUAGES CXX CUDA)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 50)
# Uncomment the line below when compiling on Jetson Nano
# set(CMAKE_CUDA_ARCHITECTURES 53)

# Set optimization level
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Add executable
add_executable(partial_sum partial_sum.cu)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
