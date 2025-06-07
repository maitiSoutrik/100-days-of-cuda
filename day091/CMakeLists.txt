cmake_minimum_required(VERSION 3.10)
project(day091_hinge_loss CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 53) # For Jetson Nano
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA error checking utilities (assuming they might be in a shared location or defined in headers)
# For now, we'll assume they are part of the .cu files or a common header.

# Static library for the core Hinge Loss logic
add_library(hinge_loss_lib STATIC
    hinge_loss.cu
    hinge_loss.cuh
)
target_include_directories(hinge_loss_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(hinge_loss_lib PRIVATE ${CUDA_LIBRARIES}) # Link against CUDA runtime

# Main executable for demonstration/benchmark
add_executable(hinge_loss_main
    hinge_loss_main.cu
)
target_link_libraries(hinge_loss_main PRIVATE hinge_loss_lib)

# Google Test
include(GoogleTest)
add_executable(hinge_loss_test
    hinge_loss_test.cu
)
target_link_libraries(hinge_loss_test PRIVATE hinge_loss_lib GTest::gtest GTest::gmock GTest::gtest_main)

gtest_discover_tests(hinge_loss_test)
