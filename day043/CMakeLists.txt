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
find_package(cudnn REQUIRED) # Find the cuDNN package

# Define the executable
add_executable(cudnn_conv_fwd cudnn_conv_fwd.cu)

# Link CUDA runtime and cuDNN libraries
target_link_libraries(cudnn_conv_fwd ${CUDA_LIBRARIES} ${CUDNN_LIBRARIES})

# Optional: Link math library if needed
# target_link_libraries(cudnn_conv_fwd m)

# Add include directories if needed (e.g., for cuDNN headers if not found automatically)
# target_include_directories(cudnn_conv_fwd PRIVATE ${CUDNN_INCLUDE_DIRS})

# Ensure the executable is placed in the expected build directory structure
set_target_properties(cudnn_conv_fwd PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/day043
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/day043
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/day043
)

install(TARGETS cudnn_conv_fwd DESTINATION bin/day043)
