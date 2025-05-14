cmake_minimum_required(VERSION 3.10)
project(day066_mfcc_extraction CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 53)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA and cuFFT
find_package(CUDA REQUIRED)
# Note: cuFFT is typically found via FindCUDA and linked using ${CUDA_CUFFT_LIBRARIES}
# For Google Test, ensure it's findable by your CMake environment.
# This might involve FetchContent in the root CMakeLists.txt or a find_package(GTest CONFIG REQUIRED)
# if GTest is installed and provides a CMake config file.
# Assuming GTest will be available (e.g., fetched by root CMakeLists.txt or installed system-wide)

# --- Library for audio features ---
add_library(audio_features_lib STATIC
    audio_features.cu
    audio_features.cuh
)
target_include_directories(audio_features_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(audio_features_lib PRIVATE
    ${CUDA_LIBRARIES}
    ${CUDA_CUFFT_LIBRARIES} # For FFT and DCT
    m # For math functions like log, cos, sin, etc.
)
set_target_properties(audio_features_lib PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)


# --- Main benchmark executable ---
add_executable(mfcc_benchmark
    mfcc_main.cu
)
target_link_libraries(mfcc_benchmark PRIVATE
    audio_features_lib
)
set_target_properties(mfcc_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)


# --- Test executable ---
# Ensure GoogleTest is found. This often means adding something like:
# include(FetchContent)
# FetchContent_Declare(
#   googletest
#   GIT_REPOSITORY https://github.com/google/googletest.git
#   GIT_TAG    release-1.12.1 # Or a more recent stable tag
# )
# FetchContent_MakeAvailable(googletest)
# Or, if GTest is installed and findable via find_package:
# find_package(GTest CONFIG REQUIRED)

add_executable(mfcc_tests
    audio_features_test.cu
)
target_link_libraries(mfcc_tests PRIVATE
    audio_features_lib
    GTest::gtest
    GTest::gmock
    GTest::gtest_main
)
set_target_properties(mfcc_tests PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

# Enable testing
include(GoogleTest)
gtest_discover_tests(mfcc_tests)

# Add an install step for the benchmark executable (optional, but good practice)
install(TARGETS mfcc_benchmark
    RUNTIME DESTINATION bin/day066
)
