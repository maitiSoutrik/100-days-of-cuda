cmake_minimum_required(VERSION 3.10)
project(day012_bfs CUDA C)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA package
find_package(CUDA REQUIRED)

# Enable C99 standard for C files
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Add the C file as a separate library to ensure it's compiled correctly
add_library(bfs_cpu_lib STATIC bfs_cpu.c)

# Make sure the CPU library is compiled with the correct include path
target_include_directories(bfs_cpu_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Add executable with CUDA files
add_executable(bfs main.cu bfs_gpu.cu)

# Link the C library to the executable
target_link_libraries(bfs PRIVATE bfs_cpu_lib)

# Set CUDA architectures - Jetson Nano has compute capability 5.3
set_target_properties(bfs PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "53")

# Install target
install(TARGETS bfs DESTINATION bin)
