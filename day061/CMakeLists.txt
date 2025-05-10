cmake_minimum_required(VERSION 3.10)
project(day061_fisher_matrix CUDA CXX)

# Set CUDA architecture for Jetson Nano
set(CMAKE_CUDA_ARCHITECTURES 53)

# Set optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable CTest
enable_testing()

# Add GoogleTest
include(GoogleTest)

# Static library for the core Fisher Matrix logic
add_library(fisher_matrix_lib STATIC
    fisher_matrix.cu
    fisher_matrix.cuh
)
target_compile_features(fisher_matrix_lib PUBLIC cxx_std_14 cuda_std_14)
target_link_libraries(fisher_matrix_lib PRIVATE CUDA::cudart) # Add other CUDA libs like cublas, curand if needed
target_include_directories(fisher_matrix_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Main executable
add_executable(fisher_main fisher_main.cu)
target_link_libraries(fisher_main PRIVATE fisher_matrix_lib)

# Test executable
add_executable(fisher_matrix_test fisher_matrix_test.cu)
target_link_libraries(fisher_matrix_test PRIVATE fisher_matrix_lib GTest::gtest GTest::gmock GTest::gtest_main)

# Discover and add tests
gtest_discover_tests(fisher_matrix_test)

# Install targets (optional, but good practice)
install(TARGETS fisher_main fisher_matrix_test
    RUNTIME DESTINATION bin
)
install(TARGETS fisher_matrix_lib
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)
