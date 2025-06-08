cmake_minimum_required(VERSION 3.10)
project(day092_elu_activation CUDA CXX)

# Set CUDA and C++ standards
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Target Jetson Nano (Compute Capability 5.3)
set(CMAKE_CUDA_ARCHITECTURES 53)

# Optimization flags (typically -O3 for release)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Find necessary CUDA package
find_package(CUDA REQUIRED)

# --- ELU Activation Library ---
# Create a static library for the core ELU logic
add_library(elu_activation_lib STATIC
    elu_activation.cu
    # elu_activation.cuh # Headers are usually included via target_include_directories
)

# Specify include directories for the library
# PUBLIC: Consumers of this library will also get these include directories
target_include_directories(elu_activation_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link CUDA libraries to the ELU library
# Using ${CUDA_LIBRARIES} which typically includes cudart
target_link_libraries(elu_activation_lib PRIVATE ${CUDA_LIBRARIES})

# --- Main Executable ---
add_executable(elu_activation_main elu_activation_main.cu)

# Link the main executable against the ELU library and CUDA libraries
target_link_libraries(elu_activation_main PRIVATE
    elu_activation_lib
    ${CUDA_LIBRARIES}
)

# --- Google Test Executable ---
# Find GoogleTest package, which is expected to be available in the build environment
# (e.g., via system install or fetched by a higher-level CMakeLists.txt)
include(GoogleTest)
if(TARGET GTest::gtest_main)
    add_executable(elu_activation_test elu_activation_test.cu)

    # Link the test executable against the ELU library, GoogleTest, and CUDA libraries
    target_link_libraries(elu_activation_test PRIVATE
        elu_activation_lib
        GTest::gtest
        GTest::gmock
        GTest::gtest_main
        ${CUDA_LIBRARIES}
    )

    # Enable testing with CTest and discover tests from GoogleTest
    include(CTest)
    enable_testing()
    gtest_discover_tests(elu_activation_test)
else()
    message(WARNING "GoogleTest (GTest::gtest_main) not found. Tests for day092_elu_activation will not be built.")
endif()
