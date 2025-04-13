cmake_minimum_required(VERSION 3.10)

project(day034_voxel_filter CUDA CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Specify the CUDA architecture for Jetson Nano
set(CMAKE_CUDA_ARCHITECTURES 53)

# Add optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Define the executable
add_executable(voxel_filter voxel_grid_filter.cu) # Renamed target as requested

# Link necessary libraries (CUDA runtime and math)
target_link_libraries(voxel_filter PRIVATE ${CUDA_LIBRARIES} m) # Use ${CUDA_LIBRARIES} consistent with Day 33

# Optional: Add include directories if needed (e.g., for helper headers)
# include_directories(include)

# Optional: Add compiler definitions if needed
# target_compile_definitions(day034_voxel_filter PRIVATE MY_MACRO=1)

# The LOCATION property might not be available at configuration time.
# Removing the debug message as it can cause issues.
# get_target_property(TARGET_LOCATION day034_voxel_filter LOCATION)
# message(STATUS "Executable target location: ${TARGET_LOCATION}")
