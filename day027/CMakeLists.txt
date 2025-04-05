cmake_minimum_required(VERSION 3.10)

project(day027_mirror_descent CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architecture for Jetson Nano
set(CMAKE_CUDA_ARCHITECTURES 53)

# Set optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Find CUDA toolkit
find_package(CUDA REQUIRED)

# Define the executable
cuda_add_executable(mirror_descent mirror_descent.cu)

# Link CUDA runtime library using the imported target (modern CMake)
# cuda_add_executable usually links cudart automatically.
# Math library 'm' is often linked automatically or implicitly included.

# Optional: Add include directories if needed (none needed for this simple example)
# target_include_directories(mirror_descent PRIVATE ${CUDA_INCLUDE_DIRS})

# Print target properties (optional, for debugging)
get_target_property(TARGET_NAME mirror_descent NAME)
get_target_property(TARGET_SOURCES mirror_descent SOURCES)
get_target_property(TARGET_LINK_LIBS mirror_descent LINK_LIBRARIES)

message(STATUS "--------------------------------------------------")
message(STATUS "Configuring Day 27: Mirror Descent")
message(STATUS "Target Name: ${TARGET_NAME}")
message(STATUS "Sources: ${TARGET_SOURCES}")
message(STATUS "Link Libraries: ${TARGET_LINK_LIBS}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Flags: ${CMAKE_CUDA_FLAGS}")
message(STATUS "--------------------------------------------------")
