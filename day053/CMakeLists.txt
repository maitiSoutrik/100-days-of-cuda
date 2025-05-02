cmake_minimum_required(VERSION 3.10)
project(day053_lstm CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 53)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Create a static library for the LSTM implementation
add_library(lstm_lib STATIC bidirectional_lstm.cu)
set_property(TARGET lstm_lib PROPERTY CXX_STANDARD 14)
set_property(TARGET lstm_lib PROPERTY CXX_STANDARD_REQUIRED YES)
# Note: No need to link CUDA_LIBRARIES to the static library itself

# Executable for main demonstration
add_executable(bidirectional_lstm main_lstm.cu)
target_link_libraries(bidirectional_lstm PRIVATE lstm_lib ${CUDA_LIBRARIES} m) # Link against the library and CUDA runtime
set_property(TARGET bidirectional_lstm PROPERTY CXX_STANDARD 14)
set_property(TARGET bidirectional_lstm PROPERTY CXX_STANDARD_REQUIRED YES)

# Add unit test
add_executable(bidirectional_lstm_test test_bidirectional_lstm.cu) # Test executable only needs its own source now
target_link_libraries(bidirectional_lstm_test PRIVATE lstm_lib ${CUDA_LIBRARIES} m gtest gtest_main pthread) # Link against the library, CUDA runtime, and gtest
set_property(TARGET bidirectional_lstm_test PROPERTY CXX_STANDARD 14)
set_property(TARGET bidirectional_lstm_test PROPERTY CXX_STANDARD_REQUIRED YES)
# Point include directories to gtest if needed, assuming gtest is found by find_package
# target_include_directories(bidirectional_lstm_test PRIVATE ${GTEST_INCLUDE_DIRS}) # Uncomment if GTEST_INCLUDE_DIRS is needed

# Enable testing for this directory
enable_testing()
add_test(NAME BidirectionalLSTMTest COMMAND bidirectional_lstm_test)
