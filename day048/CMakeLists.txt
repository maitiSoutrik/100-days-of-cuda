cmake_minimum_required(VERSION 3.10)
project(day048_aabb_collision CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 53) # Jetson Nano
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

find_package(CUDA REQUIRED)

cuda_add_executable(aabb_collision aabb_collision.cu)

target_link_libraries(aabb_collision ${CUDA_LIBRARIES} m)

message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")
message(STATUS "Project Source Dir: ${PROJECT_SOURCE_DIR}")
message(STATUS "Target aabb_collision created for Day 48.")
