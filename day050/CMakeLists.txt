cmake_minimum_required(VERSION 3.14) # Need 3.11+ for FetchContent, 3.14 recommended

project(day050_cuda_gtest LANGUAGES CUDA CXX)

# --- CUDA Configuration ---
find_package(CUDA REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 53) # Target Jetson Nano
set(CMAKE_CUDA_STANDARD 14)      # Specify CUDA C++ standard if needed
set(CMAKE_CXX_STANDARD 14)       # Specify C++ standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optional: Add optimization flags (consistent with .clinerules)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# --- Fetch Google Test ---
include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip # Use a specific, stable tag
  # Or use Git repository:
  # GIT_REPOSITORY https://github.com/google/googletest.git
  # GIT_TAG        release-1.14.0
)

# Set properties to avoid building gtest/gmock as standalone projects
# Only make GTest available after FetchContent_MakeAvailable has been called
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE) # Needed on Windows, harmless elsewhere
FetchContent_MakeAvailable(googletest)

# --- Build Test Executable ---
# cuda_add_executable handles mixing CUDA (.cu) and C++ (.cpp) sources
cuda_add_executable(vector_add_test
  vector_add_kernel.cu
  vector_add_test.cpp
)

# Link dependencies
# Note: Use ${CUDA_LIBRARIES} as specified in .clinerules
# gtest and gtest_main targets are made available by FetchContent_MakeAvailable(googletest)
target_link_libraries(vector_add_test PRIVATE
  gtest             # Google Test library
  gtest_main        # Provides main() for Google Test
  ${CUDA_LIBRARIES} # CUDA Runtime library
  m                 # Math library, if needed (often implicitly linked, but good practice)
)

# --- Testing Configuration ---
# Enable testing for this directory and subdirectories
enable_testing()

# Automatically discover tests from the executable linked with gtest_main
include(GoogleTest)
gtest_discover_tests(vector_add_test)

# --- Output ---
message(STATUS "Configuring Day 50: CUDA Unit Testing with Google Test")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Test executable: vector_add_test")
message(STATUS "Google Test Source Dir: ${googletest_SOURCE_DIR}")
message(STATUS "Google Test Binary Dir: ${googletest_BINARY_DIR}")
