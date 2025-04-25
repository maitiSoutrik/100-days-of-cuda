cmake_minimum_required(VERSION 3.10)
project(day047_dynamic_parallelism CUDA CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Specify CUDA Architecture for Jetson Nano
set(CMAKE_CUDA_ARCHITECTURES 53)

# Add optimization flags (Separable compilation handles rdc=true)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Find CUDA package to get necessary libraries
find_package(CUDA REQUIRED)

# Define the executable
add_executable(dynamic_parallelism_example dynamic_parallelism_example.cu)

# Explicitly enable separable compilation for the target (required for -rdc=true)
set_target_properties(dynamic_parallelism_example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link CUDA libraries, including the device runtime for dynamic parallelism
target_link_libraries(dynamic_parallelism_example PUBLIC ${CUDA_LIBRARIES} ${CUDA_cudadevrt_LIBRARY})

# Optional: Link math library if needed
# target_link_libraries(dynamic_parallelism_example PUBLIC m)

# Print build info
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "CUDA Arch: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Flags: ${CMAKE_CUDA_FLAGS}")
message(STATUS "CXX Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Linking CUDA Libraries: ${CUDA_LIBRARIES}")
message(STATUS "Linking CUDA Device Runtime: ${CUDA_cudadevrt_LIBRARY}")
