# metal/ - Strata's Metal (Apple Silicon) kernels, standalone for now: the runtime face (src/mtl_rt.*), the MSL
# kernels (kernels/*.metal, compiled at run time), CPU references (src/ref_*.cpp) and the parity / bench programs
# (tests/).  Nothing here is used by the CUDA / HIP / SYCL builds.
#
#   cmake -S metal -B build-metal -G Ninja -DCMAKE_BUILD_TYPE=Release && ninja -C build-metal
cmake_minimum_required(VERSION 3.20)
project(strata_metal LANGUAGES C CXX OBJCXX)

if(NOT APPLE)
  message(FATAL_ERROR "metal/ builds on macOS only")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_OBJCXX_FLAGS_RELEASE "-O2")

get_filename_component(STRATA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(GGML_COMMON "${STRATA_ROOT}/third_party/ggml/ggml-common.h")

add_library(mtl_rt STATIC src/mtl_rt.mm src/ref_quant.cpp)
target_include_directories(mtl_rt PUBLIC src ${STRATA_ROOT}/third_party/ggml)
target_compile_options(mtl_rt PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
target_compile_definitions(mtl_rt PRIVATE
  STRATA_GGML_COMMON_DEFAULT="${GGML_COMMON}"
  STRATA_METAL_KERNELS_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/kernels")
target_link_libraries(mtl_rt PUBLIC "-framework Metal" "-framework Foundation")

file(GLOB STRATA_METAL_TESTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp)
foreach(src ${STRATA_METAL_TESTS})
  get_filename_component(t ${src} NAME_WE)
  add_executable(${t} ${src})
  target_link_libraries(${t} PRIVATE mtl_rt)
endforeach()
