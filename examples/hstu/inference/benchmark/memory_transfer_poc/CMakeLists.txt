# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.18)
project(nve_memory_transfer_poc LANGUAGES CXX)

set(NVE_ROOT "" CACHE PATH "NVE source/header tree matching the linked libnve-common")
set(NVE_LIB_DIR "" CACHE PATH "Optional directory containing the matching libnve-common.so")
set(NVE_CACHE_LINE_SIZE "" CACHE STRING "Match NVE's build definition; blank uses its Linux getconf default")
if(NOT EXISTS "${NVE_ROOT}/include/gpu_table.hpp")
  message(FATAL_ERROR "Set NVE_ROOT to the NVE source tree used to build libnve-common.so")
endif()

# NVE exposes this compile definition in public headers but has no installed
# CMake target carrying it. Match its Linux default, or pass the value found in
# the library build's CMakeCache.txt / compile_commands.json explicitly.
if(NVE_CACHE_LINE_SIZE STREQUAL "")
  execute_process(COMMAND getconf LEVEL1_DCACHE_LINESIZE
    OUTPUT_VARIABLE NVE_CACHE_LINE_SIZE OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _cache_line_result)
  if(NOT _cache_line_result EQUAL 0)
    message(FATAL_ERROR "Cannot detect NVE_CACHE_LINE_SIZE; pass the value used to build libnve-common")
  endif()
endif()
if(NOT NVE_CACHE_LINE_SIZE MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "NVE_CACHE_LINE_SIZE must be the integer used to build libnve-common")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(CUDAToolkit REQUIRED)
find_library(NVE_COMMON_LIBRARY NAMES nve-common
  HINTS "${NVE_LIB_DIR}" "${NVE_ROOT}/lib" "${NVE_ROOT}/build/lib"
        "${NVE_ROOT}/python/pynve"
  NO_DEFAULT_PATH REQUIRED)
find_path(PYBIND11_INCLUDE_DIR pybind11/pybind11.h
  HINTS "${NVE_ROOT}/third_party/pybind11/include")
if(NOT PYBIND11_INCLUDE_DIR)
  execute_process(COMMAND "${Python3_EXECUTABLE}" -c
    "import pybind11; print(pybind11.get_include())"
    OUTPUT_VARIABLE _pybind_include OUTPUT_STRIP_TRAILING_WHITESPACE)
  find_path(PYBIND11_INCLUDE_DIR pybind11/pybind11.h HINTS "${_pybind_include}" REQUIRED)
endif()
find_path(DLPACK_INCLUDE_DIR dlpack/dlpack.h
  HINTS "${NVE_ROOT}/third_party/dlpack/include" "${NVE_ROOT}/include" REQUIRED)
find_path(NVE_JSON_INCLUDE_DIR nlohmann/json.hpp
  HINTS "${NVE_ROOT}/third_party/json/single_include"
        "${NVE_ROOT}/third_party/json/include" REQUIRED)

Python3_add_library(memory_transfer_poc MODULE bindings.cpp)
target_compile_features(memory_transfer_poc PRIVATE cxx_std_17)
target_compile_definitions(memory_transfer_poc PRIVATE NVE_CACHE_LINE_SIZE=${NVE_CACHE_LINE_SIZE})
target_include_directories(memory_transfer_poc PRIVATE
  "${NVE_ROOT}" "${NVE_ROOT}/include"
  "${PYBIND11_INCLUDE_DIR}" "${DLPACK_INCLUDE_DIR}" "${NVE_JSON_INCLUDE_DIR}")
target_link_libraries(memory_transfer_poc PRIVATE CUDA::cudart "${NVE_COMMON_LIBRARY}")
get_filename_component(_nve_library_directory "${NVE_COMMON_LIBRARY}" DIRECTORY)
set_target_properties(memory_transfer_poc PROPERTIES
  BUILD_RPATH "${_nve_library_directory}"
  INSTALL_RPATH "${_nve_library_directory}")
message(STATUS "POC NVE headers: ${NVE_ROOT}")
message(STATUS "POC NVE library: ${NVE_COMMON_LIBRARY}")
message(STATUS "POC NVE cache line size: ${NVE_CACHE_LINE_SIZE}")
