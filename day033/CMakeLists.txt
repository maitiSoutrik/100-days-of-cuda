cmake_minimum_required(VERSION 3.10)
project(day033_reduction_warp_shfl LANGUAGES CXX CUDA)

# Set CUDA architecture specifically for Jetson Nano
set(CMAKE_CUDA_ARCHITECTURES 53)

# Set optimization level
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Add executable
add_executable(reduction_warp_shfl reduction_warp_shfl.cu)

# Set C++ standard (using 14 for consistency with potential future C++ features)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard (consistent with C++ standard)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Link CUDA runtime library (using the variable defined by find_package(CUDA))
target_link_libraries(reduction_warp_shfl ${CUDA_LIBRARIES})

# Link math library (needed for fabs, fmaxf in main)
target_link_libraries(reduction_warp_shfl m)
