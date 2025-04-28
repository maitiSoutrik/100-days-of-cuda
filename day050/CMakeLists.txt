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

# --- Build Executables ---

# 1. Demonstration Executable (Standalone)
cuda_add_executable(vector_add_demo
  vector_add_kernel.cu
  vector_add_main.cu
)
target_link_libraries(vector_add_demo
  ${CUDA_LIBRARIES}
  m
)
message(STATUS "Demo executable: vector_add_demo")


# 2. Test Executable (Google Test)
# cuda_add_executable handles mixing CUDA (.cu) and C++ (.cpp) sources
# Note: vector_add_test.cu now used instead of .cpp to ensure NVCC compilation
cuda_add_executable(vector_add_test
  vector_add_kernel.cu # Re-use the kernel code
  vector_add_test.cu
)

# Link dependencies for the test executable
# Note: Use ${CUDA_LIBRARIES} as specified in .clinerules
# gtest and gtest_main targets are made available by FetchContent_MakeAvailable(googletest)
# Using plain signature to match potential internal linking style of CUDA libraries
target_link_libraries(vector_add_test
  gtest             # Google Test library
  gtest_main        # Provides main() for Google Test
  ${CUDA_LIBRARIES} # CUDA Runtime library (as per .clinerules & user feedback)
  m                 # Math library, if needed (often implicitly linked, but good practice)
)
message(STATUS "Test executable: vector_add_test")


# --- Testing Configuration ---
# Enable testing for this directory and subdirectories
enable_testing()

# Automatically discover tests from the executable linked with gtest_main
include(GoogleTest)
gtest_discover_tests(vector_add_test)

# --- Output ---
message(STATUS "Configuring Day 50: CUDA Demo and Unit Testing with Google Test")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Google Test Source Dir: ${googletest_SOURCE_DIR}")
message(STATUS "Google Test Binary Dir: ${googletest_BINARY_DIR}")
