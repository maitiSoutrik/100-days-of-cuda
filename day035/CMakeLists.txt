cmake_minimum_required(VERSION 3.18)
project(day035_kalman_predict CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA package
find_package(CUDA REQUIRED)

# Set CUDA architecture for Jetson Nano (Compute Capability 5.3)
set(CMAKE_CUDA_ARCHITECTURES 53)

# Set optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Add executable
add_executable(kalman_predict kalman_predict.cu)

# Find cuBLAS (part of CUDA Toolkit, linking via ${CUDA_LIBRARIES} often sufficient,
# but explicit find can be clearer if needed for specific components)
# find_package(CUBLAS) # Not strictly necessary if using ${CUDA_LIBRARIES}

# Link libraries
target_link_libraries(kalman_predict
    ${CUDA_LIBRARIES} # Links cuda_runtime and core libraries
    # If specific cuBLAS symbols are needed and not covered by CUDA_LIBRARIES:
    # ${CUDA_CUBLAS_LIBRARIES} # Prefer using CUDA_LIBRARIES if possible as per rules
    m # Link math library if needed
)

# Optional: Add include directories if necessary (e.g., for helper headers)
# target_include_directories(kalman_predict PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Message to confirm settings
message(STATUS "Configuring Day 35: Kalman Filter Prediction (cuBLAS)")
message(STATUS "Targeting CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Flags: ${CMAKE_CUDA_FLAGS}")
message(STATUS "Linking Libraries: ${CUDA_LIBRARIES}")
