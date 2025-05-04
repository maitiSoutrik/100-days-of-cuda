# day056/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(day056_mish_activation CUDA CXX)

# Set C++ standard (consistent with other days)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA package
find_package(CUDA REQUIRED)

# Find Google Test package
find_package(GTest REQUIRED)

# Set CUDA architecture for Jetson Nano (sm_53)
set(CMAKE_CUDA_ARCHITECTURES 53)

# Set optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# --- Mish Activation Library ---
# Contains core function definitions (CPU/GPU)
add_library(mish_activation_lib STATIC mish_activation_lib.cu)
# Ensure library includes headers from current dir
target_include_directories(mish_activation_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# Link common libraries needed by the library code itself
target_link_libraries(mish_activation_lib PRIVATE
    ${CUDA_LIBRARIES}
    m
)

# --- Main Benchmark Executable ---
add_executable(mish_benchmark mish_activation.cu) # Contains main()
# Link benchmark against the library
target_link_libraries(mish_benchmark PRIVATE mish_activation_lib)
# Ensure benchmark includes headers from current dir (redundant if lib propagates via PUBLIC, but safe)
target_include_directories(mish_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})


# --- Test Executable ---
enable_testing()
add_executable(mish_benchmark_test mish_activation_test.cu) # Contains test main() via GTest::gtest_main
# Link test against the library and GTest
target_link_libraries(mish_benchmark_test PRIVATE
    mish_activation_lib # Link our library
    GTest::gtest
    GTest::gtest_main
)
# Ensure test includes headers from current dir (redundant if lib propagates via PUBLIC, but safe)
target_include_directories(mish_benchmark_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Discover and add tests to CTest
include(GoogleTest)
gtest_discover_tests(mish_benchmark_test)

# Optional: Print build info
message(STATUS "Configuring Day 56: Mish Activation Library, Benchmark & Tests")
message(STATUS "Target architecture: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Flags: ${CMAKE_CUDA_FLAGS}")
message(STATUS "Found GTest: ${GTEST_FOUND}")
