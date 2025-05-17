cmake_minimum_required(VERSION 3.18)
project(day069_password_cracker CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

# Jetson Nano compute capability
set(CMAKE_CUDA_ARCHITECTURES 53)

# Optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Define the library for the core password cracking logic
add_library(password_cracker_lib STATIC
    password_cracker.cu
    password_cracker.cuh
)
target_compile_features(password_cracker_lib PUBLIC cxx_std_14)
target_link_libraries(password_cracker_lib PRIVATE ${CUDA_LIBRARIES})
target_include_directories(password_cracker_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Define the main executable
add_executable(password_cracker_main main.cu)
target_link_libraries(password_cracker_main PRIVATE password_cracker_lib)
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_libraries(password_cracker_main PRIVATE Threads::Threads m)
endif()

# Google Test configuration
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG main # Or a specific release tag like release-1.14.0
)
FetchContent_MakeAvailable(googletest)

# Define the test executable
add_executable(password_cracker_test password_cracker_test.cu)
target_link_libraries(password_cracker_test PRIVATE password_cracker_lib gtest_main) # gtest_main links gtest, gmock, and pthread
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_libraries(password_cracker_test PRIVATE Threads::Threads m)
endif()

# Discover and add tests
include(GoogleTest)
gtest_discover_tests(password_cracker_test)

# Install executables (optional, but good practice)
install(TARGETS password_cracker_main password_cracker_test
    RUNTIME DESTINATION bin
)
