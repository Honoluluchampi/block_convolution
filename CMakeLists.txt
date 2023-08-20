cmake_minimum_required(VERSION 3.16)

project(block_fft VERSION 1.0)

add_executable(block_fft block_convolution/fft_gpu.cu)

project(cuda_test)
set(TEST_SRC test/test.cpp)

add_executable(cuda_test ${TEST_SRC})
enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-g3 -O0")
target_link_libraries(cuda_test gtest gtest_main pthread)