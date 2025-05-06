cmake_minimum_required(VERSION 3.10)
project(day058_bitonic_sort CUDA CXX)

# Set CUDA architecture for Jetson Nano
set(CMAKE_CUDA_ARCHITECTURES 53)

# Set optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# Define the library for the core Bitonic Sort logic
add_library(bitonic_sort_lib STATIC
    bitonic_sort.cu
    bitonic_sort.cuh
)
target_include_directories(bitonic_sort_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# No specific CUDA libraries like cuBLAS needed for this basic sort

# Define the main executable
add_executable(bitonic_sort_main
    bitonic_sort_main.cu
)
target_link_libraries(bitonic_sort_main PRIVATE bitonic_sort_lib)

# --- Google Test ---
# Assuming Google Test is available (e.g., fetched by root CMakeLists.txt or installed system-wide)
# The root CMakeLists.txt should handle find_package(GTest REQUIRED) and include(GoogleTest)
# If not, these lines would be needed here or in the root.
# For this project, we assume the root CMake handles GTest setup.

if(TARGET GTest::gtest)
    # Define the test executable
    add_executable(bitonic_sort_test
        bitonic_sort_test.cu
    )
    target_link_libraries(bitonic_sort_test PRIVATE bitonic_sort_lib GTest::gtest GTest::gmock GTest::gtest_main)

    # Discover and add tests
    include(GoogleTest)
    gtest_discover_tests(bitonic_sort_test)
else()
    message(WARNING "Google Test not found. Skipping tests for day058_bitonic_sort.")
endif()

# Install executables (optional, but good practice for CI)
install(TARGETS bitonic_sort_main DESTINATION bin/day058)
if(TARGET bitonic_sort_test)
    install(TARGETS bitonic_sort_test DESTINATION bin/day058_tests)
endif()
