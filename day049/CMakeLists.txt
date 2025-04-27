cmake_minimum_required(VERSION 3.18) # Use a recent version compatible with OpenCV find_package
project(day049_perception_pipeline CUDA CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set CUDA architecture for Jetson Nano
set(CMAKE_CUDA_ARCHITECTURES 53)

# Set optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3") # Also optimize C++ host code

# Find CUDA Toolkit
find_package(CUDA REQUIRED)

# Find OpenCV (required for image I/O)
# Ensure OpenCV is installed in the build environment (e.g., on Jetson or CI runner)
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV or check OpenCV_DIR.")
endif()

# Include directories (CUDA and OpenCV)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

# Define the executable
add_executable(day049_perception_pipeline perception_pipeline.cu)

# Link libraries
target_link_libraries(day049_perception_pipeline
    ${CUDA_LIBRARIES}    # Link CUDA runtime
    ${OpenCV_LIBS}       # Link OpenCV libraries
    m                    # Link math library (for sqrtf)
)
