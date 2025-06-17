cmake_minimum_required(VERSION 3.18)
project(day098_mandelbrot_amr CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 53)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Enable C++11 features for CUDA, which are needed for dynamic parallelism
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++11 -rdc=true")
# Link the cudadevrt library, which is necessary for dynamic parallelism
set(CMAKE_CUDA_LIBRARIES "${CMAKE_CUDA_LIBRARIES} cudadevrt")


# Define the library for the core Mandelbrot logic
add_library(mandelbrot_amr_lib STATIC
    mandelbrot_amr.cu
    mandelbrot_amr.cuh
)
target_include_directories(mandelbrot_amr_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mandelbrot_amr_lib PRIVATE ${CUDA_LIBRARIES} ${CMAKE_CUDA_LIBRARIES})

# Define the main executable
add_executable(mandelbrot_main
    mandelbrot_main.cu
)
target_link_libraries(mandelbrot_main PRIVATE mandelbrot_amr_lib)

# Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# Define the test executable
add_executable(mandelbrot_amr_test
    mandelbrot_amr_test.cu
)
target_link_libraries(mandelbrot_amr_test PRIVATE mandelbrot_amr_lib GTest::gtest GTest::gmock GTest::gtest_main)

gtest_discover_tests(mandelbrot_amr_test)
