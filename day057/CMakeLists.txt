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

# --- Library (if we want to separate logic, optional for now) ---
# add_library(day057_cgm_cublas_lib cgm_cublas.cu cgm_cublas.cuh)
# target_link_libraries(day057_cgm_cublas_lib PUBLIC ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

# --- Main Executable ---
add_executable(cgm_benchmark cgm_cublas.cu cgm_cublas.cuh)
target_link_libraries(cgm_benchmark PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} m)
# If using the library approach:
# target_link_libraries(cgm_benchmark PRIVATE day057_cgm_cublas_lib)

# --- Testing ---
find_package(GTest REQUIRED) # Assumes GoogleTest is findable on the build system

add_executable(cgm_test cgm_cublas_test.cu cgm_cublas.cu cgm_cublas.cuh) # Include .cu for implementation details needed by test
target_link_libraries(cgm_test PRIVATE GTest::gtest GTest::gmock GTest::gtest_main ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} m)
# If using the library approach:
# target_link_libraries(cgm_test PRIVATE day057_cgm_cublas_lib GTest::gtest GTest::gmock GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(cgm_test)

# Optional: Add install rules if needed later
# install(TARGETS cgm_benchmark DESTINATION bin)
# install(TARGETS day057_cgm_cublas_lib ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
# install(FILES cgm_cublas.cuh DESTINATION include)
