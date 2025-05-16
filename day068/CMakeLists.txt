cmake_minimum_required(VERSION 3.10)
project(day068_lora CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 53) # For Jetson Nano
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add CUDA toolkit includes and libraries
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# LoRA Library (core logic, no main function)
add_library(lora_lib STATIC
    lora.cu
    lora.cuh
)
target_include_directories(lora_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# Link against CUDA runtime, cuBLAS, and math library
target_link_libraries(lora_lib PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} m)

# Main benchmark executable
add_executable(lora_benchmark
    lora_main.cu
)
target_link_libraries(lora_benchmark PRIVATE lora_lib)

# Google Test setup
include(GoogleTest)
if(TARGET GTest::gtest)
    # Test executable
    add_executable(lora_test_exec
        lora_test.cu
    )
    target_link_libraries(lora_test_exec PRIVATE lora_lib GTest::gtest GTest::gmock GTest::gtest_main)
    
    include(CTest)
    enable_testing()
    gtest_discover_tests(lora_test_exec)
else()
    message(WARNING "GoogleTest not found. Tests for day068 will not be built.")
endif()
