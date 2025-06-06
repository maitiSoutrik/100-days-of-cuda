cmake_minimum_required(VERSION 3.18)
project(day089_memory_coalescing CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 53) # Jetson Nano - Maxwell

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

add_executable(memory_coalescing_main
    memory_coalescing_main.cu
    memory_coalescing.cu
)

# Link CUDA libraries
target_link_libraries(memory_coalescing_main PRIVATE ${CUDA_LIBRARIES})

# Install the executable
install(TARGETS memory_coalescing_main
    DESTINATION bin/day089
)

# Optional: Print install location
message(STATUS "Executable will be installed to: ${CMAKE_INSTALL_PREFIX}/bin/day089")

