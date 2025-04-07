cmake_minimum_required(VERSION 3.18) # Increased to support CUDA:: namespace reliably
project(day029_k_means_assignment CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architecture for Jetson Nano
set(CMAKE_CUDA_ARCHITECTURES 53)

# Set optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")

add_executable(k_means_assignment k_means_assignment.cu)

# Link CUDA runtime and math library
target_link_libraries(k_means_assignment PRIVATE ${CUDA_LIBRARIES} m)

# Set executable output directory relative to build root
set_target_properties(k_means_assignment PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/day029
)

message(STATUS "Target 'k_means_assignment' configured for Day 29.")
message(STATUS "CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Executable will be placed in: ${CMAKE_BINARY_DIR}/day029")
