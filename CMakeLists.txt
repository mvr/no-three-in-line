cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Debug)

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")

project(three)

include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)

set(CMAKE_CUDA_ARCHITECTURES "87-real")
# set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v --generate-line-info --expt-relaxed-constexpr")

enable_testing()

add_subdirectory(test)

add_executable(three main.cpp three_kernel.cu)
