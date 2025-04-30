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

# Add the main executable target directly from its source file
cuda_add_executable(min_gru_scan min_gru_scan.cu)

# Link libraries (using plain signature for consistency)
# Using ${CUDA_LIBRARIES} as specified in .clinerules
target_link_libraries(min_gru_scan ${CUDA_LIBRARIES} m)

# Optional: Add include directories if needed (e.g., if utils were separate)
# target_include_directories(min_gru_scan PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS min_gru_scan DESTINATION bin/day052)

# --- Google Test Integration ---
if(TARGET GTest::gtest)
    # Add the test executable target, including the implementation source file
    # This compiles min_gru_scan.cu again, but links with gtest_main instead of its own main()
    cuda_add_executable(min_gru_scan_test min_gru_scan_test.cu min_gru_scan.cu)

    # Link libraries for the test executable (using plain signature for consistency)
    # IMPORTANT: Link the *implementations* needed by the tests if they aren't inline
    # In this case, the core CUDA/CPU functions are declared in min_gru_scan.h
    # and implemented in min_gru_scan.cu. We might need to link the object file
    # from min_gru_scan.cu *without* its main().
    # Let's try just linking the required libraries first. If it fails finding symbols,
    # we might need a separate object library for the shared functions *excluding* main.
    target_link_libraries(min_gru_scan_test ${CUDA_LIBRARIES} m GTest::gtest GTest::gtest_main)

    # Add test to CTest
    include(GoogleTest)
    gtest_discover_tests(min_gru_scan_test)
else()
    message(WARNING "GTest not found. Skipping tests for day052.")
endif()
