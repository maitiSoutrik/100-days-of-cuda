cmake_minimum_required(VERSION 3.10)
project(day090_frobenius_norm CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 53) # For Jetson Nano
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# CUDA files for the library
set(LIB_SOURCES
    frobenius_norm.cu
)
set(LIB_HEADERS
    frobenius_norm.cuh
)

# Add the library
add_library(frobenius_norm_lib STATIC
    ${LIB_SOURCES}
    ${LIB_HEADERS}
)
target_include_directories(frobenius_norm_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(frobenius_norm_lib PRIVATE ${CUDA_LIBRARIES})

# Main executable
add_executable(frobenius_norm_main frobenius_norm_main.cu)
target_link_libraries(frobenius_norm_main PRIVATE frobenius_norm_lib)

# Google Test
include(GoogleTest)
if(NOT TARGET GTest::gtest)
    message(WARNING "Google Test not found. Tests for day090 will not be built.")
else()
    add_executable(frobenius_norm_test frobenius_norm_test.cu)
    target_link_libraries(frobenius_norm_test PRIVATE frobenius_norm_lib GTest::gtest GTest::gmock GTest::gtest_main)
    gtest_discover_tests(frobenius_norm_test)
    message(STATUS "Added frobenius_norm_test for day090")
endif()
