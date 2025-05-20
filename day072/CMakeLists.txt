cmake_minimum_required(VERSION 3.18) # Increased to 3.18 for better GTest handling if needed
project(day072_tvd_loss LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architecture for Jetson Nano
set(CMAKE_CUDA_ARCHITECTURES 53)

# Set optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# --- Library for TVD Loss logic ---
add_library(tvd_loss_lib STATIC
    tvd_loss.cu
    tvd_loss.cuh
)
target_include_directories(tvd_loss_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# Link CUDA runtime. Using ${CUDA_LIBRARIES} as per .clinerules preference.
# For math functions like fabsf, m may be needed if not pulled by CUDA.
target_link_libraries(tvd_loss_lib PRIVATE ${CUDA_LIBRARIES} m)


# --- Main Benchmark Executable ---
add_executable(tvd_benchmark
    tvd_main.cu
)
target_link_libraries(tvd_benchmark PRIVATE tvd_loss_lib)

# --- Google Test ---
# Ensure GoogleTest is included from the top-level CMakeLists.txt or fetch it here.
# Assuming GoogleTest has been found and targets GTest::gtest etc. are available.
# If not, add:
# include(FetchContent)
# FetchContent_Declare(
#   googletest
#   GIT_REPOSITORY https://github.com/google/googletest.git
#   GIT_TAG main # Or a specific release tag
# )
# FetchContent_MakeAvailable(googletest)

# --- Test Executable ---
add_executable(tvd_loss_test
    tvd_loss_test.cu
)
target_link_libraries(tvd_loss_test PRIVATE tvd_loss_lib GTest::gtest GTest::gmock GTest::gtest_main)

# Enable testing
include(GoogleTest)
gtest_discover_tests(tvd_loss_test)

# Install executables (optional, but good practice for CI)
install(TARGETS tvd_benchmark tvd_loss_test
    RUNTIME DESTINATION ${CMAKE_PROJECT_NAME}
)
