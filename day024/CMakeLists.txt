cmake_minimum_required(VERSION 3.10)
project(day024_glu CUDA CXX)

# Find CUDA package
find_package(CUDA REQUIRED)

# Set CUDA architecture for Jetson Nano (Compute Capability 5.3)
set(CMAKE_CUDA_ARCHITECTURES 53)

# Set optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable
add_executable(glu glu.cu)

# Link CUDA libraries
target_link_libraries(glu ${CUDA_LIBRARIES} m)
