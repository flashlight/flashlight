cmake_minimum_required(VERSION 3.10)

cmake_dependent_option(FL_EXT_BUILD_HALIDE
  "Build and link Halide integration" OFF
  "FL_BUILD_CORE;FL_USE_CUDA" OFF)

if (FL_EXT_BUILD_HALIDE)
  include(${CMAKE_CURRENT_LIST_DIR}/halide/CMakeLists.txt)
endif()
