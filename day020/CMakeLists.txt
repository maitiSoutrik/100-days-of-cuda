cmake_minimum_required(VERSION 3.10)
project(monte_carlo_option_pricing CUDA CXX)

# Set CUDA architecture based on Jetson Nano
set(CMAKE_CUDA_ARCHITECTURES 53)  # Compute capability 5.3 for Jetson Nano

# Find required packages
find_package(CUDA REQUIRED)

# Set CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++14")

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})

# Add executables
add_executable(european_option_pricing
    european_option_pricing.cu
)

add_executable(asian_option_pricing
    asian_option_pricing.cu
)

# Link libraries
target_link_libraries(european_option_pricing
    ${CUDA_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    m
)

target_link_libraries(asian_option_pricing
    ${CUDA_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    m
)

# Set C++ standard
set_target_properties(european_option_pricing PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
)

set_target_properties(asian_option_pricing PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
)