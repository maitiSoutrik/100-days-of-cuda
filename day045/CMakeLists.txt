cmake_minimum_required(VERSION 3.18)
project(day045_optical_flow_gradients CUDA CXX)

# Set CUDA architecture for Jetson Nano
set(CMAKE_CUDA_ARCHITECTURES 53)

# Set optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA package
find_package(CUDA REQUIRED)

# Find OpenCV package (needed for camera input)
find_package(OpenCV REQUIRED)

# Define the executable
add_executable(optical_flow_gradients optical_flow_gradients.cu)

# Link CUDA and OpenCV libraries
# Using ${CUDA_LIBRARIES} as per .clinerules preference
target_link_libraries(optical_flow_gradients PRIVATE ${CUDA_LIBRARIES} m ${OpenCV_LIBS})

# Add include directories if needed (e.g., for OpenCV, helper headers)
target_include_directories(optical_flow_gradients PRIVATE ${OpenCV_INCLUDE_DIRS})
# target_include_directories(optical_flow_gradients PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Optional: Enable verbose output during build
# set(CMAKE_VERBOSE_MAKEFILE ON)
