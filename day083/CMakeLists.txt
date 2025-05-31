cmake_minimum_required(VERSION 3.10)
project(day083_min_reduction_dim CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 53) # For Jetson Nano
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Add include directory for the header file
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Define the library for the core reduction logic
add_library(min_reduce_lib STATIC
    min_reduce.cu
    min_reduce.cuh
)
target_link_libraries(min_reduce_lib PRIVATE ${CUDA_LIBRARIES}) # Link against CUDA runtime
target_include_directories(min_reduce_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


# Define the main executable for demonstration/benchmark
add_executable(min_reduce_main
    min_reduce_main.cu
)
target_link_libraries(min_reduce_main PRIVATE min_reduce_lib)

# Define the test executable
add_executable(min_reduce_test
    min_reduce_test.cu
)

# Link test executable against the library and Google Test
# Ensure GoogleTest is found or fetched by the root CMakeLists.txt
find_package(GTest REQUIRED)
target_link_libraries(min_reduce_test PRIVATE min_reduce_lib GTest::gtest GTest::gmock GTest::gtest_main)

# Enable testing with CTest
include(GoogleTest)
gtest_discover_tests(min_reduce_test)

# Optional: Print build information
message(STATUS "Configuring Day 083: Minimum Reduction over a Dimension")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Flags: ${CMAKE_CUDA_FLAGS}")
message(STATUS "CXX Standard: ${CMAKE_CXX_STANDARD}")
