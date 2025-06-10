cmake_minimum_required(VERSION 3.18)
project(day094_ddpm_steps CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architecture for Jetson Nano
set(CMAKE_CUDA_ARCHITECTURES 53)

# Optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# --- CUDA Kernels Library ---
add_library(ddpm_cuda_kernels STATIC
    ddpm_kernels.cu
    ddpm_kernels.cuh
)
target_include_directories(ddpm_cuda_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ddpm_cuda_kernels PRIVATE ${CUDA_LIBRARIES} m) # m for math library

# --- Main Executable (Demonstration) ---
add_executable(ddpm_demo
    ddpm_main.cu
)
target_link_libraries(ddpm_demo PRIVATE ddpm_cuda_kernels)

# --- Test Executable (Google Test) ---
include(GoogleTest)
add_executable(ddpm_tests
    ddpm_test.cu
)
target_link_libraries(ddpm_tests PRIVATE ddpm_cuda_kernels GTest::gtest GTest::gmock GTest::gtest_main)

if(TARGET GTest::gtest)
    gtest_discover_tests(ddpm_tests)
endif()

# Install targets (optional, but good practice)
install(TARGETS ddpm_demo ddpm_tests
    RUNTIME DESTINATION bin
)
install(TARGETS ddpm_cuda_kernels
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
install(FILES ddpm_kernels.cuh
    DESTINATION include
)
