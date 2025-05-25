cmake_minimum_required(VERSION 3.18) # Increased for gtest_discover_tests convenience
project(day076_contrastive_loss CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 53) # Jetson Nano
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# --- Library for Contrastive Loss ---
add_library(contrastive_loss_lib STATIC
    contrastive_loss.cu
    contrastive_loss.cuh
)
target_include_directories(contrastive_loss_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(contrastive_loss_lib PRIVATE ${CUDA_LIBRARIES})

# --- Main Executable ---
add_executable(contrastive_loss_main contrastive_loss_main.cu)
target_link_libraries(contrastive_loss_main PRIVATE contrastive_loss_lib)

# --- Google Test ---
# Ensure GoogleTest is found or fetched by the root CMakeLists.txt
# The root CMakeLists.txt should handle FetchContent for GoogleTest if not installed.
include(GoogleTest)

add_executable(contrastive_loss_test_exe contrastive_loss_test.cu)
target_link_libraries(contrastive_loss_test_exe PRIVATE
    contrastive_loss_lib
    GTest::gtest
    GTest::gmock
    GTest::gtest_main
)

# Add tests to CTest
gtest_discover_tests(contrastive_loss_test_exe)

# --- Installation (Optional, but good practice) ---
install(TARGETS contrastive_loss_main contrastive_loss_test_exe
    RUNTIME DESTINATION bin
)
install(TARGETS contrastive_loss_lib
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
install(FILES contrastive_loss.cuh DESTINATION include)

# Print a message
message(STATUS "Configured Day 76: Contrastive Loss")
