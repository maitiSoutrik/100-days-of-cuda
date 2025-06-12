cmake_minimum_required(VERSION 3.10)
project(day095_barnsley_fern CUDA CXX)

find_package(CUDA COMPONENTS curand REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 53)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Core logic library
add_library(barnsley_fern_lib STATIC
    barnsley_fern_kernel.cu
)
target_include_directories(barnsley_fern_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# The curand library is linked to the final executables that use the fern logic.
# If barnsley_fern_lib itself had host code calling cuRAND APIs directly (not just kernels using it),
# then it might need to link cuRAND. For now, kernels use it, executables link it.

# Main application
add_executable(barnsley_fern_app
    barnsley_fern_main.cu
)
target_link_libraries(barnsley_fern_app PRIVATE barnsley_fern_lib CUDA::curand)

# Google Test setup
include(GoogleTest)

add_executable(barnsley_fern_test
    barnsley_fern_test.cu
)
target_link_libraries(barnsley_fern_test PRIVATE barnsley_fern_lib GTest::gtest_main CUDA::curand)

include(CTest)
gtest_discover_tests(barnsley_fern_test)
