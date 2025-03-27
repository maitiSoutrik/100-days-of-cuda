cmake_minimum_required(VERSION 3.10)
project(day017_naive_bayes LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

add_executable(naive_bayes
  naive_bayes_kernel.cu
  main.cpp
)

target_link_libraries(naive_bayes PRIVATE CUDA::cudart)
