cmake_minimum_required(VERSION 3.10)
project(day067_swiglu CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 53)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add include directory for helper macros if any (e.g. error checking)
# For now, assume they are in the .cu files or standard.
# include_directories(${CMAKE_SOURCE_DIR}/include) # If we had a common include dir

# SwiGLU library
add_library(swiglu_lib STATIC
    swiglu.cu
    swiglu.cuh
)
target_include_directories(swiglu_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(swiglu_lib PRIVATE ${CUDA_LIBRARIES})

# Main executable for demonstration/benchmark
add_executable(swiglu_benchmark swiglu_main.cu)
target_link_libraries(swiglu_benchmark PRIVATE swiglu_lib)

# Google Test
include(GoogleTest)
if(NOT TARGET GTest::gtest)
    message(WARNING "GoogleTest not found by find_package(GTest CONFIG). Fetching from source...")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG    release-1.12.1 # Or a more recent stable release
    )
    FetchContent_MakeAvailable(googletest)
endif()

# Test executable
add_executable(swiglu_test swiglu_test.cu)
target_link_libraries(swiglu_test PRIVATE swiglu_lib GTest::gtest GTest::gmock GTest::gtest_main)

include(CTest)
enable_testing()
gtest_discover_tests(swiglu_test)

message(STATUS "Configuring Day 67: SwiGLU Operation")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Flags: ${CMAKE_CUDA_FLAGS}")
message(STATUS "CXX Standard: ${CMAKE_CXX_STANDARD}")
