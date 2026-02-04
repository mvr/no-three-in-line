cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Release)

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")

project(three)

include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)

if(CMAKE_VERSION VERSION_LESS "3.24.0")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=native")
else()
set(CMAKE_CUDA_ARCHITECTURES "native")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v --generate-line-info --expt-relaxed-constexpr")

# Allow overriding N via -DTHREE_N=19 in the cmake configure step.
if(DEFINED THREE_N)
  add_compile_definitions(THREE_N=${THREE_N})
endif()

# enable_testing()
# add_subdirectory(test)

add_executable(three main.cpp three_kernel.cu)
add_executable(three_c4 main_c4.cpp three_kernel_c4.cu three_kernel.cu)
add_executable(three_frontier main_frontier.cpp three_kernel.cu)

target_include_directories(three PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vendor ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(three_c4 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vendor ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(three_frontier PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vendor ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_compile_definitions(THREE_ENABLE_STATS)
