cmake_minimum_required(VERSION 3.18)
project(day039_thrust_basics LANGUAGES CUDA CXX)

# Set CUDA architecture for Jetson Nano
set(CMAKE_CUDA_ARCHITECTURES 53)

# Set optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA Toolkit - includes Thrust headers
find_package(CUDA REQUIRED)

# Define the executable
add_executable(thrust_examples thrust_basics.cu)

# Link CUDA runtime library
target_link_libraries(thrust_examples PRIVATE ${CUDA_LIBRARIES})

# Optional: Link math library if needed (uncomment if using functions from math.h/cmath)
# target_link_libraries(thrust_examples PRIVATE m)

# Print installation path for clarity during build
message(STATUS "Target executable: ${CMAKE_BINARY_DIR}/day039/thrust_examples")
