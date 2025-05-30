cmake_minimum_required(VERSION 3.10)
project(day080_matrix_mul_swish_scale CUDA CXX)

# Set CUDA architecture for Jetson Nano
set(CMAKE_CUDA_ARCHITECTURES 53)

# Set optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Set C++ standard (consistent with other days, e.g., C++14 or C++17)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# Include directory for headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Define the library for the core CUDA logic
add_library(swish_matrix_ops_lib STATIC
    src/swish_matrix_ops.cu
    include/swish_matrix_ops.cuh
)
target_include_directories(swish_matrix_ops_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(swish_matrix_ops_lib PRIVATE ${CUDA_LIBRARIES}) # Link CUDA runtime

# Define the main benchmark executable
add_executable(day080_benchmark
    src/benchmark_main.cu
)

# Link the benchmark executable against the library
target_link_libraries(day080_benchmark PRIVATE swish_matrix_ops_lib)

# Enable testing (even if no tests are added immediately, for consistency)
enable_testing()

# Optional: If Google Test is used later, it would be configured here.
# For now, just enabling CTest.
# Example for adding a test if a test executable 'day080_tests' is created:
# add_executable(day080_tests tests/your_test_file.cu)
# target_link_libraries(day080_tests PRIVATE swish_matrix_ops_lib GTest::gtest GTest::gtest_main)
# include(GoogleTest)
# gtest_discover_tests(day080_tests)

# Print a message indicating the build target
message(STATUS "Configuring Day 80: Matrix Multiplication with Swish Activation and Scaling")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Benchmark executable: day080_benchmark")
