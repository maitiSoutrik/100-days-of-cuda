cmake_minimum_required(VERSION 3.18)
project(day055_quantization_comparison LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

# Explicitly set CUDA architecture for Jetson Nano (sm_53)
set(CMAKE_CUDA_ARCHITECTURES 53)

# Set optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Find CUDA package (provides CUDA_LIBRARIES)
find_package(CUDA REQUIRED)

# Find GoogleTest package (needed for testing)
# Ensure GTest is installed on the build/target system or provide a path
find_package(GTest REQUIRED)

# Add the source files for kernels (can be compiled directly into executables)
set(KERNEL_SOURCES quantization_kernels.cu)

# Add the main executable
add_executable(quantization_comparison main.cu ${KERNEL_SOURCES})

# Link CUDA runtime library and math library to main executable
target_link_libraries(quantization_comparison PRIVATE ${CUDA_LIBRARIES} m)

# Ensure the main executable is built in the correct directory relative to the build root
set_target_properties(quantization_comparison PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/day055)

# --- Testing ---
# Add the test executable
add_executable(quantization_test quantization_test.cu ${KERNEL_SOURCES})

# Link CUDA runtime, math library, and Google Test to the test executable
target_link_libraries(quantization_test PRIVATE ${CUDA_LIBRARIES} m GTest::gtest GTest::gtest_main)

# Ensure the test executable is built in the correct directory relative to the build root
set_target_properties(quantization_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/day055)

# Add the test to CTest
include(GoogleTest)
gtest_add_tests(TARGET quantization_test)

# Optional: Print libraries found for debugging
# message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")
# message(STATUS "GTest Libraries: ${GTEST_LIBRARIES}; GTest Main Libraries: ${GTEST_MAIN_LIBRARIES}")
