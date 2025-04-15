cmake_minimum_required(VERSION 3.18)
project(day037_nn_forward_pass LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

# Explicitly set CUDA architecture for Jetson Nano
set(CMAKE_CUDA_ARCHITECTURES 53)

# Optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Find CUDA libraries (including cuBLAS)
find_package(CUDA REQUIRED)
if(NOT CUDA_CUBLAS_LIBRARIES)
    find_package(CUDAToolkit REQUIRED)
endif()

# Define the executable
add_executable(nn_forward_pass nn_forward_pass.cu)

# Link CUDA runtime and cuBLAS libraries
# Using CUDA_LIBRARIES for runtime as per .clinerules preference
# Using CUDA_CUBLAS_LIBRARIES for cuBLAS
target_link_libraries(nn_forward_pass PUBLIC ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

# Optional: Add include directories if needed (e.g., for helper headers)
# target_include_directories(nn_forward_pass PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS nn_forward_pass DESTINATION bin)

message(STATUS "CUDA Found: ${CUDA_FOUND}")
message(STATUS "CUDA Include Dirs: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")
message(STATUS "cuBLAS Libraries: ${CUDA_CUBLAS_LIBRARIES}")
