cmake_minimum_required(VERSION 3.10)
project(day065_geglu CUDA CXX)

# Set CUDA architecture for Jetson Nano (sm_53)
set(CMAKE_CUDA_ARCHITECTURES 53)

# Standard C++
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA specific flags (e.g., optimization)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Enable C++ exceptions for GTest, if not enabled by default for CUDA/CXX project
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr") # Might be needed for GTest with CUDA

# --- GEGLU Library ---
add_library(geglu_lib STATIC geglu.cu geglu.cuh)
target_include_directories(geglu_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# Link against CUDA runtime. Using ${CUDA_LIBRARIES} for broader compatibility as per clinerules.
# If CUDA::cudart is preferred and works consistently in the target env, it can be used.
target_link_libraries(geglu_lib PRIVATE ${CUDA_LIBRARIES})

# --- Main Benchmark Executable ---
add_executable(geglu_benchmark geglu_main.cu)
target_link_libraries(geglu_benchmark PRIVATE geglu_lib)

# --- Google Test ---
# Ensure GoogleTest is found (typically handled by the root CMakeLists.txt or CI environment)
# If not, you might need:
# include(FetchContent)
# FetchContent_Declare(
#   googletest
#   GIT_REPOSITORY https://github.com/google/googletest.git
#   GIT_TAG main # Or a specific release tag
# )
# FetchContent_MakeAvailable(googletest)
# include(GoogleTest)

# --- Test Executable ---
add_executable(geglu_test_exec geglu_test.cu)
target_link_libraries(geglu_test_exec PRIVATE geglu_lib GTest::gtest GTest::gmock GTest::gtest_main)

# Discover and add tests
include(GoogleTest)
gtest_discover_tests(geglu_test_exec)
