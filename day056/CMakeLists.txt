# day056/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(day056_mish_activation CUDA CXX)

# Set C++ standard (consistent with other days)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA package
find_package(CUDA REQUIRED)

# Find Google Test package
# Assumes GTest is installed on the system or fetched by a parent CMakeLists.txt
find_package(GTest REQUIRED)

# Set CUDA architecture for Jetson Nano (sm_53)
set(CMAKE_CUDA_ARCHITECTURES 53)

# Set optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Include the current directory for the header file
# Note: Using target_include_directories is generally preferred over include_directories
# Making it INTERFACE allows propagation if needed, PRIVATE limits it. Using PRIVATE here.

# --- Main Benchmark Executable ---
add_executable(mish_benchmark mish_activation.cu)

# Link required libraries for the benchmark
target_link_libraries(mish_benchmark PRIVATE
    ${CUDA_LIBRARIES}
    m
)
# Ensure benchmark includes headers from current dir
target_include_directories(mish_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})


# --- Test Executable ---
# Enable testing for this directory (already enabled at root, but doesn't hurt)
enable_testing()

# Add the test executable
# Include mish_activation.cu to get the definition of checkCuda, or move checkCuda to header/separate file
add_executable(mish_benchmark_test mish_activation_test.cu mish_activation.cu)

# Link required libraries for the test executable
target_link_libraries(mish_benchmark_test PRIVATE
    ${CUDA_LIBRARIES}
    m
    GTest::gtest
    GTest::gtest_main # Links the main function for gtest
)
# Ensure test includes headers from current dir and GTest
target_include_directories(mish_benchmark_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    # ${GTEST_INCLUDE_DIRS} # Often not needed when using GTest::gtest target
)

# Discover and add tests to CTest
include(GoogleTest)
gtest_discover_tests(mish_benchmark_test)

# Optional: Print build info
message(STATUS "Configuring Day 56: Mish Activation Benchmark & Tests")
message(STATUS "Target architecture: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Flags: ${CMAKE_CUDA_FLAGS}")
message(STATUS "Found GTest: ${GTEST_FOUND}")
