cmake_minimum_required(VERSION 3.10)
project(day044_occupancy_grid CUDA CXX)

# Set CUDA architecture for Jetson Nano
set(CMAKE_CUDA_ARCHITECTURES 53)

# Set optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the executable
cuda_add_executable(occupancy_grid occupancy_grid.cu)

# Link required libraries
# Need CUDA runtime and potentially math library
find_package(CUDA REQUIRED)
target_link_libraries(occupancy_grid PRIVATE ${CUDA_LIBRARIES} m)

# Optional: Add include directories if needed later
# include_directories(...)
