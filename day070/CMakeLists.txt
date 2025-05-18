cmake_minimum_required(VERSION 3.18) # Increased to 3.18 for better GTest handling if needed
project(day070_mse LANGUAGES CUDA CXX)

# Set CUDA architecture for Jetson Nano
set(CMAKE_CUDA_ARCHITECTURES 53)

# Set optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Set C++ standard (consistent with project, e.g., C++14 or C++17)
# Reverting to C++14 for broader compatibility as per typical project settings and error.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# --- MSE Library ---
# Contains the core mse.cu and mse.cuh logic
add_library(mse_lib STATIC
    mse.cu
    mse.cuh
)

# Link CUDA runtime library. Using ${CUDA_LIBRARIES} for broader compatibility.
# If specific libraries like cuBLAS were needed, they'd be added here.
target_link_libraries(mse_lib PRIVATE ${CUDA_LIBRARIES} m) # m for math library (sqrt, etc.)

# Make headers available for executables linking against this library
target_include_directories(mse_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# --- Main Benchmark Executable ---
add_executable(mse_benchmark
    mse_main.cu
)
target_link_libraries(mse_benchmark PRIVATE mse_lib)

# --- Google Test Configuration ---
# Ensure GoogleTest is found or fetched (as per root CMakeLists.txt or local setup)
# This assumes GoogleTest has been properly set up by the root CMakeLists.txt
# or is available in the system/environment.
# If FetchContent is used in the root, GTest::gtest etc. should be available.
include(GoogleTest)
if(NOT TARGET GTest::gtest)
    message(WARNING "GoogleTest targets (GTest::gtest) not found. Tests might not build correctly. Ensure GoogleTest is configured in the root CMakeLists.txt or installed.")
endif()

# --- Test Executable ---
add_executable(mse_test_runner
    mse_test.cu
)
target_link_libraries(mse_test_runner PRIVATE 
    mse_lib 
    GTest::gtest # Modern CMake target for gtest
    GTest::gmock # Modern CMake target for gmock
    GTest::gtest_main # Modern CMake target for gtest_main
)

# Add tests to CTest
if(TARGET mse_test_runner AND TARGET GTest::gtest)
    gtest_discover_tests(mse_test_runner
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
else()
    message(WARNING "mse_test_runner or GTest::gtest target not found. Cannot discover tests.")
endif()

# Optional: Print build information
message(STATUS "Configuring Day 70: Mean Squared Error")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Flags: ${CMAKE_CUDA_FLAGS}")
message(STATUS "CXX Standard: ${CMAKE_CXX_STANDARD}")
