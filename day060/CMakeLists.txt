cmake_minimum_required(VERSION 3.18)
project(day060_muon_optimization CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 53) # Jetson Nano

# Optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Define a library for the core Muon optimization logic
add_library(muon_core STATIC
    muon_optimization.cu
    muon_optimization.cuh
)
target_include_directories(muon_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(muon_core PRIVATE ${CUDA_LIBRARIES} m) # m for math library if needed

# Define the main executable
add_executable(muon_demo main.cu)
target_link_libraries(muon_demo PRIVATE muon_core)

# Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

# Define the test executable
add_executable(muon_tests muon_optimization_test.cu)
target_link_libraries(muon_tests PRIVATE muon_core GTest::gtest GTest::gmock GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(muon_tests)

install(TARGETS muon_demo DESTINATION bin)
install(TARGETS muon_tests DESTINATION bin)
