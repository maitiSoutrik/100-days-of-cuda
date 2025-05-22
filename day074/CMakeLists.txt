cmake_minimum_required(VERSION 3.10)
project(day074_rope_2d CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 53)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Set optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Define the library for the core RoPE 2D logic
add_library(rope_2d_lib STATIC
    rope_2d.cu
    rope_2d.cuh
)
target_include_directories(rope_2d_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(rope_2d_lib PRIVATE ${CUDA_LIBRARIES} m) # m for math library (sin, cos)

# Define the main executable for benchmark/demonstration
add_executable(rope_2d_benchmark
    rope_2d_main.cu
)
target_link_libraries(rope_2d_benchmark PRIVATE rope_2d_lib)

# Google Test
include(GoogleTest)
if(TARGET GTest::gtest)
    # Define the test executable
    add_executable(rope_2d_test_exec
        rope_2d_test.cu
    )
    target_link_libraries(rope_2d_test_exec PRIVATE rope_2d_lib GTest::gtest GTest::gmock GTest::gtest_main)

    # Discover and add tests
    gtest_discover_tests(rope_2d_test_exec)
else()
    message(WARNING "Google Test not found. Skipping tests for day074.")
endif()
