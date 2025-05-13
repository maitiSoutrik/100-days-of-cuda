cmake_minimum_required(VERSION 3.10)
project(day064_spectral_normalization CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 53) # Jetson Nano
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Add CUDA toolkit include directories and libraries
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Define the library for the core spectral normalization logic
add_library(spectral_norm_lib STATIC
    spectral_norm.cu
    spectral_norm.cuh
)
target_include_directories(spectral_norm_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(spectral_norm_lib PRIVATE ${CUDA_LIBRARIES} cublas ${CUDA_CURAND_LIBRARIES}) # Added cublas and curand

# Define the main application executable
add_executable(spectral_norm_app spectral_norm_main.cu)
target_link_libraries(spectral_norm_app PRIVATE spectral_norm_lib)

# --- Google Test ---
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Define the test executable
add_executable(spectral_norm_test_app spectral_norm_test.cu)
target_link_libraries(spectral_norm_test_app PRIVATE spectral_norm_lib GTest::gtest GTest::gmock GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(spectral_norm_test_app)
