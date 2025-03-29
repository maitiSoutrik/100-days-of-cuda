cmake_minimum_required(VERSION 3.10)
project(cuda_fft CUDA CXX)

# Set CUDA architecture based on Jetson Nano
set(CMAKE_CUDA_ARCHITECTURES 53)  # Compute capability 5.3 for Jetson Nano

# Find required packages
find_package(CUDA REQUIRED)
find_package(OpenCV QUIET)

# Set CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++14")

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})

# Add executable
add_executable(fft_comparison
    fft_comparison.cu
)

# Link libraries
target_link_libraries(fft_comparison
    ${CUDA_LIBRARIES}
    ${CUDA_CUFFT_LIBRARIES}
    m
)

# If OpenCV is found, build the visualization executable
# if(OpenCV_FOUND)
#     include_directories(${OpenCV_INCLUDE_DIRS})
    
#     add_executable(fft_visualization
#         fft_visualization.cu
#     )
    
#     target_link_libraries(fft_visualization
#         ${CUDA_LIBRARIES}
#         ${CUDA_CUFFT_LIBRARIES}
#         ${OpenCV_LIBS}
#         m
#     )
    
#     message(STATUS "OpenCV found, building visualization executable")
# else()
#     message(STATUS "OpenCV not found, skipping visualization executable")
# endif()

# Set C++ standard
set_target_properties(fft_comparison PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
)

# if(OpenCV_FOUND)
#     set_target_properties(fft_visualization PROPERTIES
#         CUDA_SEPARABLE_COMPILATION ON
#         CXX_STANDARD 14
#         CXX_STANDARD_REQUIRED ON
#     )
# endif()