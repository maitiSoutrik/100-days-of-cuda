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

# Add tests
find_package(GTest REQUIRED)
include(GoogleTest)
add_executable(warp_reduction_test warp_level_reduction_test.cu)
target_link_libraries(warp_reduction_test PRIVATE warp_reduction_lib GTest::GTest GTest::Main)
# For older GTest versions, it might be GTest::gtest and GTest::gtest_main
# For modern GTest with CMake, GTest::GTest and GTest::Main is common.
# The root CMakeLists.txt uses googletest_add_test with ALIAS GTest::gtest etc.
# Let's try to be consistent or use what FetchContent provides.
# The root CMakeLists.txt does `FetchContent_MakeAvailable(googletest)` which should make GTest::𝒕𝒂𝒓𝒈𝒆𝒕𝒔 available.
# The standard targets are GTest::GTest (for the library) and GTest::Main (for GTest's main).
# If using gtest_discover_tests, you often link to GTest::GTest and GTest::gtest_main (or GTest::Main)

# Using gtest_discover_tests is preferred for modern CMake
gtest_discover_tests(warp_reduction_test)

install(TARGETS warp_reduction_main DESTINATION bin)
