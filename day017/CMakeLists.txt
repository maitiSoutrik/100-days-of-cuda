cmake_minimum_required(VERSION 3.18)
project(day017_naive_bayes LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(naive_bayes
  naive_bayes_kernel.cu
  main.cpp
)

set_property(TARGET naive_bayes PROPERTY CUDA_ARCHITECTURES 53)
target_link_libraries(naive_bayes PRIVATE CUDA::cudart)
