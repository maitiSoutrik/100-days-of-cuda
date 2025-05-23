cmake_minimum_required(VERSION 3.10)
project(day075_fused_linear_softmax_loss LANGUAGES CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 53)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Define the library for the core logic
add_library(fused_linear_softmax_loss_lib STATIC
    fused_linear_softmax_loss.cu
    fused_linear_softmax_loss.cuh
)
target_link_libraries(fused_linear_softmax_loss_lib PRIVATE ${CUDA_LIBRARIES})
target_include_directories(fused_linear_softmax_loss_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Define the main application executable
add_executable(fused_linear_softmax_loss_app main.cu)
target_link_libraries(fused_linear_softmax_loss_app PRIVATE fused_linear_softmax_loss_lib)

# Google Test setup
# The GoogleTest targets (GTest::gtest, GTest::gmock, GTest::gtest_main)
# are available because FetchContent_MakeAvailable(googletest) was called in the root CMakeLists.txt
include(GoogleTest)

# Define the test executable
add_executable(fused_linear_softmax_loss_test_app fused_linear_softmax_loss_test.cu)
target_link_libraries(fused_linear_softmax_loss_test_app PRIVATE
    fused_linear_softmax_loss_lib
    GTest::gtest
    GTest::gmock
    GTest::gtest_main
)

# Discover and add tests
gtest_discover_tests(fused_linear_softmax_loss_test_app)
