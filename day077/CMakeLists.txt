cmake_minimum_required(VERSION 3.10)
project(day077_huber_loss CUDA CXX)

# Set CUDA architecture for Jetson Nano
set(CMAKE_CUDA_ARCHITECTURES 53)

# Set optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Define the library for Huber Loss
add_library(huber_loss_lib STATIC huber_loss.cu huber_loss.cuh)
target_link_libraries(huber_loss_lib PRIVATE ${CUDA_LIBRARIES})
target_include_directories(huber_loss_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Define the main executable
add_executable(huber_loss_main huber_loss_main.cu)
target_link_libraries(huber_loss_main PRIVATE huber_loss_lib)

# Define the test executable
add_executable(huber_loss_test huber_loss_test.cu)
target_link_libraries(huber_loss_test PRIVATE huber_loss_lib GTest::gtest GTest::gmock GTest::gtest_main)

# Enable testing
include(GoogleTest)
gtest_discover_tests(huber_loss_test)
