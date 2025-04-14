cmake_minimum_required(VERSION 3.10)

# Set the project name
project(day036_cusparse_spmv CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard (matches C++ standard)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture for Jetson Nano (sm_53)
set(CMAKE_CUDA_ARCHITECTURES 53)

# Set optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Find CUDA toolkit and explicitly request the cusparse component
find_package(CUDA REQUIRED COMPONENTS cusparse)

# Add executable
add_executable(cusparse_spmv cusparse_spmv.cu)

# Link against CUDA runtime and cuSPARSE libraries
# Use ${CUDA_LIBRARIES} for runtime (as per rules) and try the modern target for cusparse
target_link_libraries(cusparse_spmv PRIVATE ${CUDA_LIBRARIES} CUDA::cusparse)

# Link math library (often needed)
target_link_libraries(cusparse_spmv PRIVATE m)
