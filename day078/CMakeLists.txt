cmake_minimum_required(VERSION 3.10)
project(day078_dyt CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 53)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# CUDA files for the library
set(DYT_LIB_SRCS
    dyt_core.cu
    dyt.cuh
)

# Add the library for core DyT logic
add_library(dyt_lib STATIC ${DYT_LIB_SRCS})
target_include_directories(dyt_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dyt_lib PRIVATE ${CUDA_LIBRARIES})

# Add the main executable for demonstration
add_executable(dyt_demo dyt_main.cu)
target_link_libraries(dyt_demo PRIVATE dyt_lib)

# Google Test setup
# The root CMakeLists.txt should handle FetchContent for googletest
# Ensure GTest::gtest, GTest::gmock, GTest::gtest_main targets are available
include(GoogleTest)

# Add the test executable
add_executable(dyt_test dyt_test.cu)
target_link_libraries(dyt_test PRIVATE dyt_lib GTest::gtest GTest::gmock GTest::gtest_main)

# Discover and add tests to CTest
gtest_discover_tests(dyt_test)
