cmake_minimum_required(VERSION 3.10)
project(day079_upper_tri_gemm CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 53)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CXX_STANDARD 14)

add_executable(upper_tri_gemm upper_tri_gemm_main.cu upper_tri_gemm.cu)
target_link_libraries(upper_tri_gemm PRIVATE ${CUDA_LIBRARIES})

add_library(upper_tri_gemm_lib STATIC upper_tri_gemm.cu upper_tri_gemm.cuh)
target_link_libraries(upper_tri_gemm_lib PRIVATE ${CUDA_LIBRARIES})
target_include_directories(upper_tri_gemm_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(upper_tri_gemm_test upper_tri_gemm_test.cu)
target_link_libraries(upper_tri_gemm_test PRIVATE upper_tri_gemm_lib GTest::gtest GTest::gmock GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(upper_tri_gemm_test)
