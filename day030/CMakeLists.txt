cmake_minimum_required(VERSION 3.10)
project(day030_camera_intensity CUDA CXX)

# Set C++ standard (consistent with previous days)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architecture for Jetson Nano
set(CMAKE_CUDA_ARCHITECTURES 53)

# Set optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Find OpenCV package
# Make sure OpenCV is installed on the Jetson Nano build environment
# (e.g., sudo apt-get install libopencv-dev)
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV required but not found. Please install libopencv-dev.")
endif()

# Include directories (CUDA and OpenCV)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable
add_executable(camera_avg_intensity camera_avg_intensity.cu)

# Link libraries
# - CUDA runtime library
# - Math library (may be needed implicitly by OpenCV or CUDA)
# - OpenCV libraries
target_link_libraries(camera_avg_intensity
    PRIVATE
    ${CUDA_LIBRARIES} # Equivalent to CUDA::cudart
    m
    ${OpenCV_LIBS}
)

# Optional: Add installation rule if needed for deployment
# install(TARGETS camera_avg_intensity DESTINATION bin)

message(STATUS "Configured Day 30: Camera Average Intensity (CUDA + OpenCV)")
message(STATUS "  - OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "  - OpenCV libraries: ${OpenCV_LIBS}")
