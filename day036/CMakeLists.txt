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

# Find CUDA toolkit and its components, including cusparse
# Note: FindCUDA should find cuSPARSE automatically if CUDA toolkit is found correctly.
# However, explicitly finding the component can be clearer if issues arise.
find_package(CUDA REQUIRED) 

# Add executable
add_executable(cusparse_spmv cusparse_spmv.cu)

# Link against CUDA runtime and cuSPARSE libraries
# Using the variables provided by FindCUDA as per .clinerules
# ${CUDA_LIBRARIES} typically includes cudart
# ${CUDA_CUSPARSE_LIBRARIES} is the variable for cuSPARSE
target_link_libraries(cusparse_spmv PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUSPARSE_LIBRARIES})

# Link math library (often needed)
target_link_libraries(cusparse_spmv PRIVATE m)
