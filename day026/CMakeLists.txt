cmake_minimum_required(VERSION 3.10)
project(day026_kde CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 53)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_STANDARD 14)

add_executable(kde kde.cu)
target_link_libraries(kde PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(kde PRIVATE m)
