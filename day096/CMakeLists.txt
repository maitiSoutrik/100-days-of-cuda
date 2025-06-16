cmake_minimum_required(VERSION 3.10)
project(day096_product_reduction_dim CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 53) # Jetson Nano
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Include current directory so headers are found
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Library containing core kernel/wrapper
add_library(product_reduction_lib STATIC
    product_reduction.cu
    product_reduction.cuh)

target_link_libraries(product_reduction_lib PRIVATE ${CUDA_LIBRARIES})

target_include_directories(product_reduction_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Demo executable
add_executable(product_reduction_main product_reduction_main.cu)
target_link_libraries(product_reduction_main PRIVATE product_reduction_lib)

# Unit tests
add_executable(product_reduction_test product_reduction_test.cu)
find_package(GTest REQUIRED)
target_link_libraries(product_reduction_test PRIVATE product_reduction_lib GTest::gtest GTest::gmock GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(product_reduction_test)

message(STATUS "Configuring Day 096: Product Reduction over a Dimension")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
