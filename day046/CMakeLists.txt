cmake_minimum_required(VERSION 3.18) # Increased slightly for better CUDA/cuBLAS handling if needed
project(day046_backpropagation_fc CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Explicitly set target architecture for Jetson Nano
set(CMAKE_CUDA_ARCHITECTURES 53)

# Set optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Find CUDA and required components (cuBLAS)
# find_package(CUDA REQUIRED) # Included by project() call with CUDA language
find_package(CUDAToolkit REQUIRED) # Modern way to ensure libraries are found

# Define the executable
add_executable(backpropagation_fc backpropagation_fc.cu)

# Link necessary libraries
# Link CUDA runtime library and math library
# Use modern CUDA:: targets if possible and reliable, otherwise fallback
# target_link_libraries(backpropagation_fc PRIVATE CUDA::cudart m) # Prefer modern targets

# As per .clinerules, avoiding CUDA::cudart due to observed inconsistencies
# Link cuBLAS using the variable provided by FindCUDAToolkit/FindCUDA
target_link_libraries(backpropagation_fc PRIVATE
    ${CUDA_LIBRARIES}    # Rule: Use this for runtime
    CUDA::cublas         # Modern target for cuBLAS
    m                    # For math functions like fabs
)

# Optional: Print libraries found for debugging
# message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")
# message(STATUS "cuBLAS Libraries: ${CUBLAS_LIBRARIES}") # Check if CUDA::cublas works

# Ensure the executable is placed in the expected directory for CI
set_target_properties(backpropagation_fc PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/day046
)

install(TARGETS backpropagation_fc DESTINATION bin/day046)
