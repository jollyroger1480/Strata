# metal/ - the Apple Silicon (Metal) backend, MAC-PLAN.md.  Its own CMake project like sycl/, macOS only, never part of
# the CUDA / HIP / SYCL builds.  Host sources are taken from ../src UNCHANGED (the point of Phase 1: shared, not
# copied): include/metal_compat goes first on the include path and gives them the CUDA runtime names on Metal, as
# include/strata/hip_compat does on HIP.  Only the GPU runtime (src/gpu) and the kernels are Metal's own.
#
#   cmake -S metal -B build-metal -G Ninja -DMETAL_CPP_DIR=/path/to/metal-cpp && cmake --build build-metal
#   ./build-metal/runtime_hello
cmake_minimum_required(VERSION 3.24)
project(strata_metal LANGUAGES CXX)

if(NOT APPLE)
  message(FATAL_ERROR "metal/ builds on macOS only")
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(STRATA_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(METAL_CPP_DIR "" CACHE PATH "metal-cpp headers (https://developer.apple.com/metal/cpp/), e.g. metal-cpp_26")
if(NOT EXISTS "${METAL_CPP_DIR}/Metal/Metal.hpp")
  message(FATAL_ERROR "METAL_CPP_DIR='${METAL_CPP_DIR}' has no Metal/Metal.hpp")
endif()

find_library(METAL_FW Metal REQUIRED)
find_library(FOUNDATION_FW Foundation REQUIRED)
find_library(QUARTZ_FW QuartzCore REQUIRED)

# the device runtime (include/strata/gpu/runtime.hpp on metal-cpp)
add_library(strata_gpu_metal STATIC src/gpu/runtime.cpp)
target_include_directories(strata_gpu_metal BEFORE PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/metal_compat
                           ${CMAKE_CURRENT_SOURCE_DIR}/include ${STRATA_ROOT}/include)
target_include_directories(strata_gpu_metal SYSTEM PUBLIC ${METAL_CPP_DIR})
target_compile_definitions(strata_gpu_metal PUBLIC STRATA_GPU_METAL=1)
target_link_libraries(strata_gpu_metal PUBLIC ${METAL_FW} ${FOUNDATION_FW} ${QUARTZ_FW})

# the shared host files built for Metal so far, taken from ../src unchanged
add_library(strata_core_metal STATIC ${STRATA_ROOT}/src/core/graph.cpp)
target_link_libraries(strata_core_metal PUBLIC strata_gpu_metal)

add_executable(runtime_hello tests/runtime_hello.cpp)
target_link_libraries(runtime_hello PRIVATE strata_core_metal)
enable_testing()
add_test(NAME runtime_hello COMMAND runtime_hello)
