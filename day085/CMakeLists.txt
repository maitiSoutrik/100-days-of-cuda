cmake_minimum_required(VERSION 3.18) # Increased to 3.18 for gtest_discover_tests convenience
project(day085_tensor_matrix_mult CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA Architectures for Jetson Nano (sm_53)
set(CMAKE_CUDA_ARCHITECTURES 53)

# Optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# --- Library for core tensor_matrix_mult logic ---
add_library(tensor_matrix_mult_lib STATIC
    tensor_matrix_mult.cu
    tensor_matrix_mult.cuh
)
target_include_directories(tensor_matrix_mult_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# target_link_libraries(tensor_matrix_mult_lib PRIVATE CUDA::cudart) # Modern CMake
# Using CUDA_LIBRARIES for broader compatibility as per rules, if CUDA::cudart is problematic
target_link_libraries(tensor_matrix_mult_lib PRIVATE ${CUDA_LIBRARIES})


# --- Main executable ---
add_executable(tensor_matrix_mult_main tensor_matrix_mult_main.cu)
target_link_libraries(tensor_matrix_mult_main PRIVATE tensor_matrix_mult_lib)
set_target_properties(tensor_matrix_mult_main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# --- Google Test executable ---
# Ensure GoogleTest is found (typically handled by root CMakeLists.txt or system install)
# For per-directory, we might need to find_package(GTest REQUIRED) if not handled globally.
# Assuming GTest is available as per project setup.
include(GoogleTest)

add_executable(tensor_matrix_mult_test tensor_matrix_mult_test.cu)
target_link_libraries(tensor_matrix_mult_test PRIVATE tensor_matrix_mult_lib GTest::gtest GTest::gmock GTest::gtest_main)
set_target_properties(tensor_matrix_mult_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(GTEST_FOUND)
    gtest_discover_tests(tensor_matrix_mult_test)
else()
    message(WARNING "GoogleTest not found. Tests for day085 will not be configured.")
endif()

# Optional: Add install targets if needed for deployment
# install(TARGETS tensor_matrix_mult_main DESTINATION bin)
# install(FILES tensor_matrix_mult.cuh DESTINATION include)
# install(TARGETS tensor_matrix_mult_lib ARCHIVE DESTINATION lib)
