cmake_minimum_required(VERSION 3.18) # Using a reasonably modern version for FetchContent
project(day082_negative_cosine_similarity CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 53) # For Jetson Nano compatibility
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3") # Optimization

# --- GoogleTest ---
# Using FetchContent to ensure GoogleTest is available
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1 # Using a specific stable tag for reproducibility
)
# Set FETCHCONTENT_QUIET to FALSE to see download progress (optional)
# set(FETCHCONTENT_QUIET FALSE) 
FetchContent_MakeAvailable(googletest)
# --- End GoogleTest ---


# CUDA Static Library for the core logic
# This library contains the kernel and its launcher, without a main function.
add_library(neg_cosine_sim_lib STATIC
    negative_cosine_similarity.cu
    negative_cosine_similarity.cuh
)
target_include_directories(neg_cosine_sim_lib PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR} # To find negative_cosine_similarity.cuh
)
# Link against CUDA runtime libraries. Using ${CUDA_LIBRARIES} is robust.
target_link_libraries(neg_cosine_sim_lib PRIVATE ${CUDA_LIBRARIES})


# Main executable for demonstration/benchmark
add_executable(neg_cosine_sim_main negative_cosine_similarity_main.cu)
target_link_libraries(neg_cosine_sim_main PRIVATE 
    neg_cosine_sim_lib      # Link against our CUDA library
    ${CUDA_LIBRARIES}       # For CUDA API calls in main (e.g., cudaMalloc)
)

# Test executable
# Check if GTest targets are available (they should be after FetchContent_MakeAvailable)
if(TARGET GTest::gtest AND TARGET GTest::gtest_main)
    add_executable(neg_cosine_sim_test negative_cosine_similarity_test.cu)
    target_link_libraries(neg_cosine_sim_test PRIVATE 
        neg_cosine_sim_lib      # Link against our CUDA library
        GTest::gtest            # GoogleTest library
        GTest::gtest_main       # GoogleTest main for running tests
        ${CUDA_LIBRARIES}       # For CUDA API calls in tests
    )
    
    # Enable testing and discover tests
    enable_testing()
    include(GoogleTest)
    gtest_discover_tests(neg_cosine_sim_test)
else()
    message(WARNING "GoogleTest targets (GTest::gtest, GTest::gtest_main) not found. Tests for day082 will not be built.")
endif()

# Optional: Installation directives (useful for CI or packaging)
# As per .clinerules, executables are expected in build/dayXXX/
# CMake's default build location for executables will be something like
# build/day082/neg_cosine_sim_main and build/day082/neg_cosine_sim_test
# so explicit installation to bin/day082 might be redundant if CI directly uses build dir.
# However, if a specific "install" step is used by CI:
# install(TARGETS neg_cosine_sim_main neg_cosine_sim_test
#     RUNTIME DESTINATION ${CMAKE_PROJECT_NAME}/bin 
#     # LIBRARY DESTINATION ${CMAKE_PROJECT_NAME}/lib
#     # ARCHIVE DESTINATION ${CMAKE_PROJECT_NAME}/lib
# )
# For this project, direct use from build directory is typical.
