cmake_minimum_required(VERSION 3.18)
project(day018_matrix_multiplication_cublas LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(matrix_multiplication_cublas matrix_multiplication_cublas.cu)

set_property(TARGET matrix_multiplication_cublas PROPERTY CUDA_ARCHITECTURES 53)
target_link_libraries(matrix_multiplication_cublas PRIVATE CUDA::cudart CUDA::cublas)