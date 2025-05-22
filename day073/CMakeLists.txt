cmake_minimum_required(VERSION 3.10)
project(day073_rope_embedding CUDA CXX)

# Set CUDA architecture for Jetson Nano (sm_53)
set(CMAKE_CUDA_ARCHITECTURES 53)

# Set optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Set C++ standard (e.g., C++14 or C++17)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# --- RoPE Embedding Library ---
# Contains the core RoPE CUDA implementation (kernels, device functions)
# This library MUST NOT contain a main function.
add_library(rope_embedding_lib STATIC
    rope_embedding.cu
    rope_embedding.cuh
)

# Link the library against necessary CUDA components
# Using ${CUDA_LIBRARIES} for cudart
target_link_libraries(rope_embedding_lib PRIVATE ${CUDA_LIBRARIES} m) # m for math library (powf, cosf, sinf)

# Make headers available
target_include_directories(rope_embedding_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


# --- Main Benchmark/Application Executable ---
# Contains only the main function and related setup/teardown/printing code.
add_executable(rope_benchmark_main rope_main.cu)

# Link the executable against the static library
target_link_libraries(rope_benchmark_main PRIVATE rope_embedding_lib)


# --- Test Executable ---
# Enable testing
enable_testing()
include(GoogleTest) # CMake should find GoogleTest (often fetched by root CMakeLists.txt)

# Define the test executable target
# The source file (rope_embedding_test.cu) contains only the Google Test code.
add_executable(rope_embedding_gtest rope_embedding_test.cu)

# Link the test executable against the static library and Google Test
target_link_libraries(rope_embedding_gtest PRIVATE
    rope_embedding_lib
    GTest::gtest
    GTest::gmock
    GTest::gtest_main # Or GTest::Main if using older GTest CMake integration
)

# Discover and add tests to CTest
gtest_discover_tests(rope_embedding_gtest)

# Optional: Print a message
message(STATUS "Configured Day 73: 1D RoPE Embedding")
