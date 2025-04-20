cmake_minimum_required(VERSION 3.18)
project(day041_radix_sort CUDA CXX)

# Set CUDA architecture for Jetson Nano
set(CMAKE_CUDA_ARCHITECTURES 53)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# Optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Find CUDA package
find_package(CUDA REQUIRED)

# Add executable
add_executable(radix_sort radix_sort.cu)

# Link CUDA libraries
target_link_libraries(radix_sort ${CUDA_LIBRARIES} m)

# Optional: If using Thrust explicitly (though scan might be custom)
# target_link_libraries(radix_sort ${CUDA_LIBRARIES} m CUDA::thrust)

# Print target properties (for debugging)
get_target_property(TARGET_NAME radix_sort NAME)
get_target_property(TARGET_SOURCES radix_sort SOURCES)
get_target_property(TARGET_LINK_LIBS radix_sort LINK_LIBRARIES)

message(STATUS "Configuring Day 41: Radix Sort")
message(STATUS "  Target: ${TARGET_NAME}")
message(STATUS "  Sources: ${TARGET_SOURCES}")
message(STATUS "  CUDA Arch: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Link Libraries: ${TARGET_LINK_LIBS}")
