cmake_minimum_required(VERSION 3.10)
project(day053_lstm CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 53)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

add_executable(bidirectional_lstm main_lstm.cu) # Use a separate file for the main executable

target_link_libraries(bidirectional_lstm PRIVATE ${CUDA_LIBRARIES} m)

set_property(TARGET bidirectional_lstm PROPERTY CXX_STANDARD 14)
set_property(TARGET bidirectional_lstm PROPERTY CXX_STANDARD_REQUIRED YES)

# Add unit test
add_executable(bidirectional_lstm_test test_bidirectional_lstm.cu)
target_link_libraries(bidirectional_lstm_test PRIVATE ${CUDA_LIBRARIES} m gtest gtest_main pthread) # Link with gtest and pthread
set_property(TARGET bidirectional_lstm_test PROPERTY CXX_STANDARD 14)
set_property(TARGET bidirectional_lstm_test PROPERTY CXX_STANDARD_REQUIRED YES)
# Point include directories to gtest if needed, assuming gtest is found by find_package
# target_include_directories(bidirectional_lstm_test PRIVATE ${GTEST_INCLUDE_DIRS}) # Uncomment if GTEST_INCLUDE_DIRS is needed

# Enable testing for this directory
enable_testing()
add_test(NAME BidirectionalLSTMTest COMMAND bidirectional_lstm_test)
