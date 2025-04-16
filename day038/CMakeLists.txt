cmake_minimum_required(VERSION 3.18) # Increased version for potentially better CUDA handling
project(day038_batch_norm_forward LANGUAGES CUDA CXX)

find_package(CUDA REQUIRED)

# Set CUDA architecture for Jetson Nano (sm_53)
set(CMAKE_CUDA_ARCHITECTURES 53) 

# Set optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Set C++ standard (optional but good practice)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 14) # Match C++ standard if using C++ features in .cu
set(CMAKE_CUDA_STANDARD_REQUIRED True)


# Add executable
# Use cuda_add_executable for simplicity
cuda_add_executable(batch_norm_forward batch_norm_forward.cu)

# Link CUDA runtime library and math library
# Using ${CUDA_LIBRARIES} as specified in .clinerules
# Using plain signature to match internal FindCUDA usage
target_link_libraries(batch_norm_forward ${CUDA_LIBRARIES} m)

# Optional: Print information during configuration
message(STATUS "Configuring Day 38: Batch Normalization Forward Pass")
message(STATUS "Target CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Executable: batch_norm_forward")
