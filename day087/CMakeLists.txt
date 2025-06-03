cmake_minimum_required(VERSION 3.18)
project(day087_softplus_activation CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 53) # Jetson Nano Target

# Optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Add the CUDA toolkit include directories and libraries
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_libraries(${CUDA_LIBRARIES})

# Create a static library for the core Softplus activation logic
add_library(softplus_lib STATIC
    softplus_activation.cu
    softplus_activation.cuh
)

# Specify include directory for the library
target_include_directories(softplus_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link CUDA runtime library to the static library
target_link_libraries(softplus_lib PRIVATE ${CUDA_LIBRARIES})

# Add the main executable
add_executable(softplus_activation_main
    softplus_activation_main.cu
)

# Link the main executable against the Softplus library
target_link_libraries(softplus_activation_main PRIVATE softplus_lib)

# Add test executable
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
add_executable(softplus_activation_test softplus_activation_test.cu)
target_link_libraries(softplus_activation_test PRIVATE softplus_lib GTest::gtest GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(softplus_activation_test)

# Print a message
message(STATUS "Configuring Day 87: Softplus Activation")
message(STATUS "Targeting CUDA Architecture: sm_53 (Jetson Nano)")
message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "CXX Standard: ${CMAKE_CXX_STANDARD}")