cmake_minimum_required(VERSION 3.10)

project(day009 CUDA)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add executable
add_executable(flash_attention flash_attention.cu)

# Set architecture flags
set_target_properties(flash_attention PROPERTIES CUDA_ARCHITECTURES "50;60;70;75")

# Add include directories if needed
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Set optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
