cmake_minimum_required(VERSION 3.18)
project(day062_batched_l2_norm CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 53) # Jetson Nano
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Define the library for the core logic
add_library(batched_l2_norm_lib STATIC
    batched_l2_norm.cu
    batched_l2_norm.cuh
)
target_include_directories(batched_l2_norm_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(batched_l2_norm_lib PRIVATE CUDA::cudart)

# Define the main executable
add_executable(batched_l2_norm_main main.cu)
target_link_libraries(batched_l2_norm_main PRIVATE batched_l2_norm_lib)

# Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

# Define the test executable
add_executable(batched_l2_norm_test batched_l2_norm_test.cu)
target_link_libraries(batched_l2_norm_test PRIVATE batched_l2_norm_lib GTest::gtest GTest::gmock GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(batched_l2_norm_test)

# Install targets (optional, but good practice)
install(TARGETS batched_l2_norm_main batched_l2_norm_test
        RUNTIME DESTINATION bin)
install(TARGETS batched_l2_norm_lib
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)
install(FILES batched_l2_norm.cuh DESTINATION include)
