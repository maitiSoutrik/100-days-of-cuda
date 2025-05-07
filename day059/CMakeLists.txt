cmake_minimum_required(VERSION 3.10)
project(day059_ray_tracing CUDA CXX)

# Set CUDA architecture for Jetson Nano
set(CMAKE_CUDA_ARCHITECTURES 53)

# Set optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Ensure CUDA_LIBRARIES is available
find_package(CUDA REQUIRED)

# Add common error checking utilities if they exist at a project level
# For now, assume they are included directly or not used in this simple example

# --- Library for core ray tracing logic ---
add_library(ray_tracer_lib STATIC
    ray_tracer.cu
    ray_tracer.cuh
)
target_include_directories(ray_tracer_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# Link CUDA math library (for sqrtf, fmaxf etc.)
target_link_libraries(ray_tracer_lib PRIVATE ${CUDA_LIBRARIES} m) # Using ${CUDA_LIBRARIES} for broader compatibility

# --- Main executable ---
add_executable(ray_tracer_main main.cu)
target_link_libraries(ray_tracer_main PRIVATE ray_tracer_lib)

# --- Google Test ---
# Ensure GoogleTest is found (typically handled by the root CMakeLists.txt or CI environment)
# If not, you might need:
# include(FetchContent)
# FetchContent_Declare(
#   googletest
#   GIT_REPOSITORY https://github.com/google/googletest.git
#   GIT_TAG main # Or a specific release tag
# )
# FetchContent_MakeAvailable(googletest)
# include(GoogleTest)

# --- Test executable ---
add_executable(ray_tracer_test ray_tracer_test.cu)
target_link_libraries(ray_tracer_test PRIVATE ray_tracer_lib GTest::gtest GTest::gmock GTest::gtest_main)

# Discover and add tests
include(GoogleTest)
gtest_discover_tests(ray_tracer_test)

# Create output directory for the PPM image
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/output)
# For running from build dir, main.cu will save to "output.ppm"
# If running from source dir after build, it will be in day059/output.ppm
# The CI script usually runs from build/dayXXX/
# Let's ensure the main.cu saves to a path relative to executable or a fixed path like CMAKE_CURRENT_BINARY_DIR
# For simplicity, the current save_image saves to "output.ppm" in the CWD of execution.
# The CI will run it from build/day059/, so output.ppm will be there.
# If we want it in day059/output/, main.cu needs to be smarter or take output path.
# For now, let's stick to the reference code's "output.ppm".
# The .clinerules mentions dayXXX/output/ for storing images.
# The save_image function in main.cu will be modified to save to "output/output.ppm".
# The CMakeLists.txt in dayXXX should create dayXXX/output if it doesn't exist.
# However, write_to_file in main.cu will create it.
# Let's ensure the output directory for the executable is clear.
# The executable will run from build/day059.
# So, if main.cu writes to "output/output.ppm", it will be build/day059/output/output.ppm
# This is fine.
