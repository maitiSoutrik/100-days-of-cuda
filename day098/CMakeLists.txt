cmake_minimum_required(VERSION 3.18)
project(day098_mandelbrot_amr CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 53)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Enable C++11 features for CUDA, which are needed for dynamic parallelism
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true") # CMAKE_CUDA_STANDARD handles the --std flag
# Link the cudadevrt library, which is necessary for dynamic parallelism
# (cudadevrt will be linked directly to the target)

# Define an OBJECT library for the core Mandelbrot logic
add_library(mandelbrot_amr_objlib OBJECT
    mandelbrot_amr.cu
    # Headers are not part of OBJECT libraries sources, they are included via include_directories
)
# No target_link_libraries for OBJECT libraries.
# Include directories will be set on consuming targets.

# Define the main executable
add_executable(mandelbrot_main
    mandelbrot_main.cu
    $<TARGET_OBJECTS:mandelbrot_amr_objlib>
)
set_property(TARGET mandelbrot_main PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_include_directories(mandelbrot_main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mandelbrot_main PRIVATE cudadevrt ${CUDA_LIBRARIES})

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
    $<TARGET_OBJECTS:mandelbrot_amr_objlib>
)
set_property(TARGET mandelbrot_amr_test PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_include_directories(mandelbrot_amr_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mandelbrot_amr_test PRIVATE GTest::gtest GTest::gmock GTest::gtest_main cudadevrt ${CUDA_LIBRARIES})

gtest_discover_tests(mandelbrot_amr_test)
