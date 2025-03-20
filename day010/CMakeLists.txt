cmake_minimum_required(VERSION 3.10)

# Set the project name
project(day010 CUDA)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add executable
add_executable(sparse_matrix_vector_mul sparse_matrix_vector_mul.cu)

# Set the CUDA architectures
set_target_properties(sparse_matrix_vector_mul PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "5.3" # For Jetson Nano compatibility
)

# Link against CUDA libraries
target_link_libraries(sparse_matrix_vector_mul PRIVATE cuda cudart)
