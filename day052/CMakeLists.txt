cmake_minimum_required(VERSION 3.18) # Use a reasonably modern version

project(day052_min_gru_scan CUDA CXX)

# Set C++ Standard (e.g., C++14, match other days)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# Set CUDA Architecture for Jetson Nano
set(CMAKE_CUDA_ARCHITECTURES 53)

# Set Optimization Flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Find CUDA
find_package(CUDA REQUIRED)
find_package(GTest REQUIRED) # Find Google Test

# Create an object library from the main source file
# This allows linking its compiled code into both the main executable and the test executable
# without compiling it twice.
add_library(min_gru_scan_lib OBJECT min_gru_scan.cu)

# Add the main executable target, linking the object library
add_executable(min_gru_scan $<TARGET_OBJECTS:min_gru_scan_lib>)

# Link libraries
# Using ${CUDA_LIBRARIES} as specified in .clinerules
target_link_libraries(min_gru_scan PRIVATE ${CUDA_LIBRARIES} m)

# Optional: Add include directories if needed (e.g., if utils were separate)
# target_include_directories(min_gru_scan PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS min_gru_scan DESTINATION bin/day052)

# --- Google Test Integration ---
if(TARGET GTest::gtest)
    # Add the test executable target, linking the object library and GTest
    cuda_add_executable(min_gru_scan_test min_gru_scan_test.cu $<TARGET_OBJECTS:min_gru_scan_lib>)

    # Link libraries for the test executable
    target_link_libraries(min_gru_scan_test PRIVATE ${CUDA_LIBRARIES} m GTest::gtest GTest::gtest_main)

    # Add test to CTest
    include(GoogleTest)
    gtest_discover_tests(min_gru_scan_test)
else()
    message(WARNING "GTest not found. Skipping tests for day052.")
endif()
