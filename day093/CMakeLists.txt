cmake_minimum_required(VERSION 3.10)
project(day093_rms_norm CUDA CXX)

# Set CUDA architecture for Jetson Nano
set(CMAKE_CUDA_ARCHITECTURES 53)

# Set optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_STANDARD 14)

# Find required packages
find_package(CUDA REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Create static library for core RMS normalization functionality
add_library(rms_norm_lib STATIC
    rms_norm.cu
    rms_norm.cuh
)

# Link CUDA libraries to the static library
target_link_libraries(rms_norm_lib PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} m)

# Make headers available
target_include_directories(rms_norm_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Create main benchmark executable
add_executable(rms_norm_benchmark rms_norm_main.cu)
target_link_libraries(rms_norm_benchmark PRIVATE rms_norm_lib)

# Create test executable
add_executable(rms_norm_test rms_norm_test.cu)
target_link_libraries(rms_norm_test PRIVATE rms_norm_lib GTest::gtest GTest::gmock GTest::gtest_main)

# Enable testing
include(GoogleTest)
gtest_discover_tests(rms_norm_test)
