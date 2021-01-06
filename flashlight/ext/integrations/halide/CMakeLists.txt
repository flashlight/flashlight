cmake_minimum_required(VERSION 3.10)

if (NOT FL_USE_CUDA)
  message(FATAL_ERROR "Flashlight Halide integration "
    "only available with the CUDA backend for now")
endif()

find_package(Halide CONFIG REQUIRED)

include(${CMAKE_MODULE_PATH}/HalideUtils.cmake)

# TODO: relax the requirement to link with libHalide if
# possible. This may not be given that we're hard-linking
# against weak symbols from the lib
target_link_libraries(
  flashlight
  PUBLIC
  Halide::Halide
  Halide::Runtime
  Halide::Tools)

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/HalideInterface.cpp
  )

# Right now, we unfortunately need to link to a libcuda stub to get Driver API
# so as to interact with the Halide nvptx runtime with needed CUcontexts.
# TODO(jacobkahn): figure out the right way to install Halide code
target_link_libraries(flashlight PUBLIC $<BUILD_INTERFACE:${CUDA_CUDA_LIBRARY}>)
# Headers for compiled pipelines
target_include_directories(flashlight PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
