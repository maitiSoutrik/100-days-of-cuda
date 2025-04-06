cmake_minimum_required(VERSION 3.10)

project(day028_mini_batch_sgd CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architecture for Jetson Nano
set(CMAKE_CUDA_ARCHITECTURES 53)

# Set optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Find CUDA toolkit
find_package(CUDA REQUIRED)

# Define the executable
cuda_add_executable(mini_batch_sgd mini_batch_sgd.cu)

# Link CUDA runtime library using the imported target (modern CMake)
# cuda_add_executable usually links cudart automatically.
# Math library 'm' is often linked automatically or implicitly included.
target_link_libraries(mini_batch_sgd m)

# Print target properties (optional, for debugging)
get_target_property(TARGET_NAME mini_batch_sgd NAME)
get_target_property(TARGET_SOURCES mini_batch_sgd SOURCES)
get_target_property(TARGET_LINK_LIBS mini_batch_sgd LINK_LIBRARIES)

message(STATUS "--------------------------------------------------")
message(STATUS "Configuring Day 28: Mini-Batch SGD for Linear Regression")
message(STATUS "Target Name: ${TARGET_NAME}")
message(STATUS "Sources: ${TARGET_SOURCES}")
message(STATUS "Link Libraries: ${TARGET_LINK_LIBS}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Flags: ${CMAKE_CUDA_FLAGS}")
message(STATUS "--------------------------------------------------")
