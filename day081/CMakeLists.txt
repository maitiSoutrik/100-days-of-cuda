cmake_minimum_required(VERSION 3.18)
project(day081_jsd_loss CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 53) # Jetson Nano
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Define the library for the JSD loss implementation
add_library(jsd_loss_lib STATIC
    jsd_loss.cu
    jsd_loss.cuh
)
target_include_directories(jsd_loss_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(jsd_loss_lib PRIVATE ${CUDA_LIBRARIES})

# Define the main executable for benchmarking
add_executable(jsd_benchmark jsd_main.cu)
target_link_libraries(jsd_benchmark PRIVATE jsd_loss_lib ${CUDA_LIBRARIES})

# Google Test
include(GoogleTest)
if(NOT TARGET GTest::gtest)
    message(STATUS "Google Test not found by include(GoogleTest). Fetching from source.")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG main # Or a specific release tag e.g., release-1.10.0
    )
    FetchContent_MakeAvailable(googletest)
    # Ensure GTest::gtest, GTest::gmock, GTest::gtest_main targets are available
    if(NOT TARGET GTest::gtest)
        get_target_property(gtest_sources googletest SOURCE_DIR)
        add_library(GTest::gtest IMPORTED GLOBAL) # Or however googletest creates its targets
        # This part might need adjustment based on how FetchContent exposes GTest targets
        # For modern CMake, GTest::gtest should be available after FetchContent_MakeAvailable
    endif()
endif()

# Define the test executable
add_executable(jsd_loss_test jsd_loss_test.cu)
target_link_libraries(jsd_loss_test PRIVATE jsd_loss_lib GTest::gtest GTest::gmock GTest::gtest_main ${CUDA_LIBRARIES})

# Discover and add tests
if(TARGET jsd_loss_test AND TARGET GTest::gtest)
    gtest_discover_tests(jsd_loss_test)
else()
    message(WARNING "jsd_loss_test or GTest::gtest target not found. Tests will not be discovered.")
endif()

install(TARGETS jsd_benchmark jsd_loss_test
    RUNTIME DESTINATION bin
)
