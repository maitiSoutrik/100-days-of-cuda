# day086/CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(day086_hard_sigmoid CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 53) # Jetson Nano

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Include path for common_utils.h and hard_sigmoid.cuh
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# CUDA Library for Hard Sigmoid
# Note: The current hard_sigmoid.cu directly implements hard_sigmoid_solution
# which includes memory management. For a library, typically the core kernel
# would be separate, and the solution function might take device pointers.
# However, following the current structure of hard_sigmoid.cu:
add_library(hard_sigmoid_lib STATIC
    hard_sigmoid.cu
    hard_sigmoid.cuh
    common_utils.h
)
target_link_libraries(hard_sigmoid_lib PRIVATE ${CUDA_LIBRARIES} m)
# No explicit target_include_directories needed for hard_sigmoid_lib itself if headers are in CMAKE_CURRENT_SOURCE_DIR
# but good practice if headers were in a dedicated 'include' subdir.

# Main Executable
add_executable(hard_sigmoid_main hard_sigmoid_main.cu)
target_link_libraries(hard_sigmoid_main PRIVATE hard_sigmoid_lib)

# Google Test for Hard Sigmoid
find_package(GTest REQUIRED)
if(GTEST_FOUND)
    include(GoogleTest)
    add_executable(hard_sigmoid_test hard_sigmoid_test.cu)
    target_link_libraries(hard_sigmoid_test PRIVATE hard_sigmoid_lib GTest::gtest GTest::gmock GTest::gtest_main)
    gtest_discover_tests(hard_sigmoid_test)
else()
    message(WARNING "Google Test not found. Tests for day086 will not be built.")
endif()
