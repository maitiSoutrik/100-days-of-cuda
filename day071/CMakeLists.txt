cmake_minimum_required(VERSION 3.18) # Increased to 3.18 for better GTest handling if needed
project(day071_group_norm_forward CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 53) # Jetson Nano
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please ensure it is installed.")
endif()

# Define the library for group normalization logic
add_library(group_norm_lib STATIC
    group_norm_forward.cu
    group_norm_forward.cuh
)
target_include_directories(group_norm_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(group_norm_lib PRIVATE ${CUDA_LIBRARIES} m) # m for math library (sqrtf etc.)

# Define the main executable
add_executable(day071_group_norm_main group_norm_main.cu)
target_link_libraries(day071_group_norm_main PRIVATE group_norm_lib ${OpenCV_LIBS})

# Google Test configuration
include(GoogleTest)
if(NOT TARGET GTest::gtest)
    message(FATAL_ERROR "Google Test not found. Please ensure it is installed or fetched by the root CMakeLists.txt")
endif()

# Define the test executable
add_executable(day071_group_norm_test group_norm_forward_test.cu)
target_link_libraries(day071_group_norm_test PRIVATE group_norm_lib GTest::gtest GTest::gmock GTest::gtest_main)

# Add tests to CTest
include(CTest)
add_test(NAME Day71_GroupNormTest COMMAND day071_group_norm_test)
