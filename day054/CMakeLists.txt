cmake_minimum_required(VERSION 3.18)
project(day054_adahessian CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Explicitly set target architecture for Jetson Nano
set(CMAKE_CUDA_ARCHITECTURES 53)

# Set optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Find CUDA package to get libraries
find_package(CUDA REQUIRED)

# Define the source file containing the kernel implementation
set(KERNEL_SOURCE adahessian.cu)

# Add the main executable
add_executable(main_adahessian main_adahessian.cu ${KERNEL_SOURCE})
target_link_libraries(main_adahessian PUBLIC ${CUDA_LIBRARIES} m)
message(STATUS "Adding executable: main_adahessian")

# Add the test executable
add_executable(test_adahessian test_adahessian.cu ${KERNEL_SOURCE})
target_link_libraries(test_adahessian PUBLIC ${CUDA_LIBRARIES} m)
message(STATUS "Adding executable: test_adahessian")

# Optional: Print message indicating the target architecture
message(STATUS "Configuring Day 54 AdaHessian for CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
