cmake_minimum_required(VERSION 3.18) # Using a slightly higher version common in newer setups
project(day043_cudnn_conv_fwd CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Explicitly set architecture for Jetson Nano
set(CMAKE_CUDA_ARCHITECTURES 53)

# Optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

find_package(CUDA REQUIRED)

# Manually find cuDNN header - more robust if CMake config is missing
find_path(CUDNN_INCLUDE_DIR NAMES cudnn_version.h # Check for a version header
                        HINTS ${CUDA_TOOLKIT_ROOT_DIR}/include
                        DOC "Path to cuDNN include directory")
if(NOT CUDNN_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find cudnn_version.h. Ensure cuDNN is installed correctly (e.g., in ${CUDA_TOOLKIT_ROOT_DIR}/include) or manually set CUDNN_INCLUDE_DIR.")
else()
    message(STATUS "Found cuDNN include directory: ${CUDNN_INCLUDE_DIR}")
endif()

# Manually find cuDNN library - more robust if CMake config is missing
find_library(CUDNN_LIBRARY NAMES cudnn # CMake searches for libcudnn.so, cudnn.lib etc.
                         HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 # Common on 64-bit Linux
                               ${CUDA_TOOLKIT_ROOT_DIR}/lib   # Fallback path
                         DOC "Path to cuDNN library")
if(NOT CUDNN_LIBRARY)
    message(FATAL_ERROR "Could not find cuDNN library (e.g., libcudnn.so). Ensure cuDNN is installed correctly (e.g., in ${CUDA_TOOLKIT_ROOT_DIR}/lib64) or manually set CUDNN_LIBRARY.")
else()
     message(STATUS "Found cuDNN library: ${CUDNN_LIBRARY}")
endif()

# Define the executable
add_executable(cudnn_conv_fwd cudnn_conv_fwd.cu)

# Add include directory for cuDNN headers
target_include_directories(cudnn_conv_fwd PRIVATE ${CUDNN_INCLUDE_DIR})

# Link CUDA runtime and the explicitly found cuDNN library
target_link_libraries(cudnn_conv_fwd PRIVATE ${CUDA_LIBRARIES} ${CUDNN_LIBRARY})

# Optional: Link math library if needed
# target_link_libraries(cudnn_conv_fwd PRIVATE m)

# Ensure the executable is placed in the expected build directory structure
set_target_properties(cudnn_conv_fwd PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/day043
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/day043
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/day043
)

install(TARGETS cudnn_conv_fwd DESTINATION bin/day043)
