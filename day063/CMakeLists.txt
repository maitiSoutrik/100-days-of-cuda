cmake_minimum_required(VERSION 3.10)
project(day063_mcl_localization CXX CUDA) # Changed language order

set(CMAKE_CUDA_ARCHITECTURES 53)
# Pass -std=c++14 to the host compiler (e.g., g++) via nvcc's -Xcompiler flag
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -std=c++14") 
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# Removing CMAKE_CUDA_STANDARD and CMAKE_CUDA_STANDARD_REQUIRED
# to rely solely on the -Xcompiler flag for host C++ standard.

# Add GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

# Core MCL Localization Library
add_library(mcl_localization_lib STATIC
    mcl_localization.cu
)
target_include_directories(mcl_localization_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mcl_localization_lib PRIVATE ${CUDA_LIBRARIES} m) # m for math library if needed

# Main Executable
add_executable(mcl_main mcl_main.cu)
target_link_libraries(mcl_main PRIVATE mcl_localization_lib)

# Test Executable
add_executable(mcl_localization_test mcl_localization_test.cu)
target_link_libraries(mcl_localization_test PRIVATE mcl_localization_lib GTest::gtest GTest::gmock GTest::gtest_main)
gtest_discover_tests(mcl_localization_test)

# Install targets (optional, but good practice)
install(TARGETS mcl_main mcl_localization_test
    RUNTIME DESTINATION bin
)
install(TARGETS mcl_localization_lib
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
install(FILES mcl_localization.cuh DESTINATION include)
