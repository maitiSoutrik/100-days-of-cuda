cmake_minimum_required(VERSION 3.10)
project(day040_texture_interpolation CUDA CXX)

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
find_package(OpenCV REQUIRED core imgproc imgcodecs videoio) # Added videoio for VideoCapture
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV (core, imgproc, imgcodecs, videoio) required but not found. Please install libopencv-dev.")
endif()

# Include directories (CUDA and OpenCV)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable
add_executable(texture_interpolation texture_interpolation.cu)

# Link libraries
# - CUDA runtime library
# - Math library (may be needed implicitly)
# - OpenCV libraries
target_link_libraries(texture_interpolation
    PRIVATE
    ${CUDA_LIBRARIES} # Equivalent to CUDA::cudart
    m
    ${OpenCV_LIBS}
)

# Copy the input image required by the executable to the build directory
# Source path relative to the project root
set(INPUT_IMAGE_SOURCE ${CMAKE_SOURCE_DIR}/day014/lena_gray.png)
# Destination path in the build directory for this day
set(INPUT_IMAGE_DEST ${CMAKE_CURRENT_BINARY_DIR}/lena_gray.png)

add_custom_command(
    TARGET texture_interpolation POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${INPUT_IMAGE_SOURCE}
            ${INPUT_IMAGE_DEST}
    COMMENT "Copying lena_gray.png to build directory for runtime access"
    VERBATIM # Ensures command is executed exactly as specified
)

message(STATUS "Configured Day 40: Texture Memory Interpolation (CUDA + OpenCV)")
message(STATUS "  - OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "  - OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "  - Input image will be copied to: ${INPUT_IMAGE_DEST}")
