cmake_minimum_required(VERSION 3.10)

# Project definition for Day 42
project(day042_n_body_optimized CUDA CXX)

# Find CUDA package
find_package(CUDA REQUIRED)

# Set CUDA architecture for Jetson Nano (CC 5.3)
set(CMAKE_CUDA_ARCHITECTURES 53)

# Set optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Set C++ standard (optional, but good practice)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14) # Match C++ standard if needed
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add executable target
add_executable(n_body_optimized n_body_optimized.cu)

# Link CUDA runtime library (Using the variable, not the target)
target_link_libraries(n_body_optimized ${CUDA_LIBRARIES})

# Link math library if needed (sqrtf is used)
target_link_libraries(n_body_optimized m)

# Print target architecture
message(STATUS "Targeting CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Flags: ${CMAKE_CUDA_FLAGS}")
message(STATUS "CXX Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "Linking Libraries: ${CUDA_LIBRARIES}")
