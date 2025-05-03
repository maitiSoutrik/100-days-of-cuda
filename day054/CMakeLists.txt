cmake_minimum_required(VERSION 3.18)
project(day054_adahessian CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Explicitly set target architecture for Jetson Nano
set(CMAKE_CUDA_ARCHITECTURES 53)

# Set optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Find CUDA package to get libraries
find_package(CUDA REQUIRED)

# Define the source file containing the kernel implementation
set(KERNEL_SOURCE adahessian.cu)

# Add the main executable
add_executable(main_adahessian main_adahessian.cu ${KERNEL_SOURCE})
target_link_libraries(main_adahessian PUBLIC ${CUDA_LIBRARIES} m)
message(STATUS "Adding executable: main_adahessian")

# Add the test executable using Google Test
# Assuming GTest is found by CMake (either installed or via parent project)
find_package(GTest REQUIRED) # Make sure GTest is found

add_executable(test_adahessian test_adahessian.cu ${KERNEL_SOURCE})
# Link against kernel implementation, CUDA runtime, math lib, and GTest targets
# Using modern GTest targets (GTest::gtest, GTest::gtest_main) is preferred
target_link_libraries(test_adahessian PRIVATE ${CUDA_LIBRARIES} m GTest::gtest GTest::gtest_main)

# Include GTest headers path if needed (modern targets often handle this)
# target_include_directories(test_adahessian PRIVATE ${GTEST_INCLUDE_DIRS}) # Usually not needed with modern targets

message(STATUS "Adding Google Test executable: test_adahessian")

# Integrate with CTest (requires enable_testing() in root CMakeLists.txt)
include(GoogleTest)
gtest_discover_tests(test_adahessian) # Automatically discover tests from the executable

# Optional: Print message indicating the target architecture
message(STATUS "Configuring Day 54 AdaHessian for CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
