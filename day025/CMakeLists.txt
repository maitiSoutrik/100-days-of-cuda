cmake_minimum_required(VERSION 3.10)
project(day025_passthrough_filter CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

# Target Jetson Nano
set(CMAKE_CUDA_ARCHITECTURES 53)

# Optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Add executable
add_executable(passthrough_filter passthrough_filter.cu)

# Link CUDA runtime library
target_link_libraries(passthrough_filter PRIVATE CUDA::cudart)

# Link math library if needed (e.g., for sqrt, pow)
target_link_libraries(passthrough_filter PRIVATE m)

# Add include directories if necessary (e.g., for helper headers)
# target_include_directories(passthrough_filter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS passthrough_filter DESTINATION bin)
