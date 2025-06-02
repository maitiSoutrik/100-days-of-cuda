cmake_minimum_required(VERSION 3.10)
project(day084_cumulative_product CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 53) # Jetson Nano
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Enable CTest
enable_testing()

# Find CUDA
find_package(CUDA REQUIRED)

# Find GoogleTest
find_package(GTest REQUIRED)

# --- Library for core cumulative product logic ---
add_library(cumulative_product_lib STATIC
    cumulative_product.cu
    cumulative_product.cuh
)
target_include_directories(cumulative_product_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cumulative_product_lib PRIVATE ${CUDA_LIBRARIES}) # Link against CUDA runtime

# --- Main executable for demonstration/benchmark ---
add_executable(cumulative_product_main cumulative_product_main.cu)
target_link_libraries(cumulative_product_main PRIVATE cumulative_product_lib ${CUDA_LIBRARIES})

# --- Test executable ---
add_executable(cumulative_product_test cumulative_product_test.cu)
target_link_libraries(cumulative_product_test PRIVATE cumulative_product_lib GTest::gtest GTest::gmock GTest::gtest_main ${CUDA_LIBRARIES})

# Discover tests
include(GoogleTest)
gtest_discover_tests(cumulative_product_test)

# Add executables to CTest
add_test(NAME Day084_Main_Test COMMAND cumulative_product_main)
# Note: For tests run via ctest, the actual GTest executable (cumulative_product_test) will handle pass/fail.
# The add_test for cumulative_product_main is more for ensuring it runs without crashing.
# Proper test validation for cumulative_product_main would be via its output or return code.

# Installation (optional, but good practice)
install(TARGETS cumulative_product_main cumulative_product_test
    RUNTIME DESTINATION bin
)
install(TARGETS cumulative_product_lib
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
install(FILES cumulative_product.cuh DESTINATION include)
