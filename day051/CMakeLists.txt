cmake_minimum_required(VERSION 3.10)
project(day051_tensorrt_onnx_infer CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 53)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Find TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS /usr/include/x86_64-linux-gnu /usr/include /usr/local/include /usr/src/tensorrt/include
)
find_library(TENSORRT_LIBRARIES nvinfer
    PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib /usr/src/tensorrt/lib
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARIES)
    message(FATAL_ERROR "TensorRT not found. Please install TensorRT and set TENSORRT_INCLUDE_DIR and TENSORRT_LIBRARIES.")
endif()

include_directories(${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR})

add_executable(onnx_infer onnx_infer.cpp)
target_link_libraries(onnx_infer ${CUDA_LIBRARIES} ${TENSORRT_LIBRARIES})

# If you use other TensorRT libs (e.g., nvinfer_plugin), add them here
