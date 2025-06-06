cmake_minimum_required(VERSION 3.18)
project(day088_warp_level_reduction CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CUDA_ARCHITECTURES 53) # Jetson Nano - Maxwell

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Define the library for the core CUDA logic
add_library(warp_reduction_lib STATIC
    warp_level_reduction.cu
    warp_level_reduction.cuh
)

target_include_directories(warp_reduction_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# No explicit linking to CUDA::cudart needed if using modern CMake with project(LANGUAGES CUDA)
# However, if issues arise, uncommenting the line below might be necessary for older CMake versions or specific setups.
# target_link_libraries(warp_reduction_lib PRIVATE CUDA::cudart)

# Define the main executable
add_executable(warp_reduction_main
    warp_level_reduction_main.cu
)

target_link_libraries(warp_reduction_main PRIVATE warp_reduction_lib)

# Optional: Add tests if you create a _test.cu file later
# find_package(GTest REQUIRED)
# include(GoogleTest)
# add_executable(warp_reduction_test warp_level_reduction_test.cu)
# target_link_libraries(warp_reduction_test PRIVATE warp_reduction_lib GTest::gtest GTest::gtest_main)
# gtest_discover_tests(warp_reduction_test)

install(TARGETS warp_reduction_main DESTINATION bin)
