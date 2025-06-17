cmake_minimum_required(VERSION 3.18)
project(day097_fused_rmsnorm_swiglu CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 53) # NVIDIA Jetson Nano
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Add relative path to GTest source if not installed system-wide
# This assumes GoogleTest is in a subdirectory of the root project or fetched by the root CMakeLists.txt
# If GoogleTest is handled by the root CMakeLists.txt, these lines might not be needed here.
# include(FetchContent)
# FetchContent_Declare(
#   googletest
#   GIT_REPOSITORY https://github.com/google/googletest.git
#   GIT_TAG main # Or a specific release tag
# )
# FetchContent_MakeAvailable(googletest)

# CUDA Utility (if it were a library, not just a header)
# add_library(cuda_utils_lib STATIC cuda_utils.h)
# target_include_directories(cuda_utils_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Core library for the fused kernel
add_library(fused_rmsnorm_swiglu_lib STATIC
    fused_rmsnorm_swiglu.cu
    fused_rmsnorm_swiglu.cuh
    cuda_utils.h
)
target_link_libraries(fused_rmsnorm_swiglu_lib PRIVATE ${CUDA_LIBRARIES} m)
target_include_directories(fused_rmsnorm_swiglu_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Main executable
add_executable(fused_rmsnorm_swiglu_main fused_rmsnorm_swiglu_main.cu)
target_link_libraries(fused_rmsnorm_swiglu_main PRIVATE fused_rmsnorm_swiglu_lib)

# Test executable
enable_testing()
include(GoogleTest)
add_executable(fused_rmsnorm_swiglu_test fused_rmsnorm_swiglu_test.cu)
target_link_libraries(fused_rmsnorm_swiglu_test PRIVATE fused_rmsnorm_swiglu_lib GTest::gtest GTest::gmock GTest::gtest_main)

gtest_discover_tests(fused_rmsnorm_swiglu_test)

install(TARGETS fused_rmsnorm_swiglu_main DESTINATION bin)
install(TARGETS fused_rmsnorm_swiglu_test DESTINATION bin)
