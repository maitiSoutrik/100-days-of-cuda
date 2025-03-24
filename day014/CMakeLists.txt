cmake_minimum_required(VERSION 3.10)

# Find CUDA package
find_package(CUDA REQUIRED)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Set CUDA flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
cuda_add_executable(fhd_image fhd_image.cu)

# Link libraries
target_link_libraries(fhd_image ${OpenCV_LIBS})
