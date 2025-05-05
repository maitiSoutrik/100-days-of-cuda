cmake_minimum_required(VERSION 3.18)
project(day057_cgm_cublas LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

# Explicitly set target architecture for Jetson Nano
set(CMAKE_CUDA_ARCHITECTURES 53)

# Optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Find CUDA and required libraries (cuBLAS)
find_package(CUDA REQUIRED)
# Note: find_package(CUDA) typically finds cuBLAS as well.
# We link against ${CUDA_LIBRARIES} and ${CUDA_CUBLAS_LIBRARIES}

# --- Shared Library ---
# Compile the core CGM logic into a static library
add_library(cgm_cublas_lib STATIC cgm_cublas.cu cgm_cublas.cuh)
# Link the library against required CUDA components
target_link_libraries(cgm_cublas_lib PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} m)
# Ensure headers from this directory are available
target_include_directories(cgm_cublas_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


# --- Main Executable ---
# The benchmark executable now only needs its own main() logic,
# but since main() is currently in cgm_cublas.cu, we still list it here.
# A better approach might be to move main() to a separate file (e.g., benchmark_main.cu)
# For now, we keep it simple:
add_executable(cgm_benchmark cgm_cublas.cu) # main() is in cgm_cublas.cu
# Link the benchmark against the library
target_link_libraries(cgm_benchmark PRIVATE cgm_cublas_lib)


# --- Testing ---
find_package(GTest REQUIRED) # Assumes GoogleTest is findable on the build system

# Test executable only includes the test source file
add_executable(cgm_test cgm_cublas_test.cu)
# Link the test against the library and GTest
target_link_libraries(cgm_test PRIVATE cgm_cublas_lib GTest::gtest GTest::gmock GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(cgm_test)

# Optional: Add install rules if needed later
# install(TARGETS cgm_benchmark DESTINATION bin)
# install(TARGETS day057_cgm_cublas_lib ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
# install(FILES cgm_cublas.cuh DESTINATION include)
