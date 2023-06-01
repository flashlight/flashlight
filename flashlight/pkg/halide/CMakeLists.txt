cmake_minimum_required(VERSION 3.16)

if (NOT (FL_USE_ARRAYFIRE AND FL_ARRAYFIRE_USE_CUDA))
  message(FATAL_ERROR "Flashlight Halide integration "
    "only available with the ArrayFire backend with CUDA for now")
endif()

add_library(fl_pkg_halide)

# Install lib and headers
list(APPEND INSTALLABLE_TARGETS fl_pkg_halide)
setup_install_headers(
  ${CMAKE_CURRENT_LIST_DIR}
  ${FL_INSTALL_INC_DIR_HEADER_LOC}/pkg)

find_package(Halide CONFIG REQUIRED)

include(${PROJECT_SOURCE_DIR}/cmake/HalideUtils.cmake)

# TODO: relax the requirement to link with libHalide if
# possible. This may not be given that we're hard-linking
# against weak symbols from the lib
target_link_libraries(
  fl_pkg_halide
  PUBLIC
  flashlight
  Halide::Halide
  Halide::Runtime
  Halide::Tools)

target_sources(
  fl_pkg_halide
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/HalideInterface.cpp
  )

# Right now, we unfortunately need to link to a libcuda stub to get Driver API
# so as to interact with the Halide nvptx runtime with needed CUcontexts.
# TODO(jacobkahn): figure out the right way to install Halide code
target_link_libraries(
  fl_pkg_halide
  PUBLIC
  $<BUILD_INTERFACE:${CUDA_CUDA_LIBRARY}>)
# Headers for compiled pipelines
target_include_directories(
  fl_pkg_halide
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

set(DIR ${CMAKE_CURRENT_LIST_DIR})
set(LIBS fl_pkg_halide)
build_test(SRC ${DIR}/test/HalideTest.cpp LIBS ${LIBS})
fl_add_and_link_halide_lib(
  SRC ${DIR}/test/HalideTestPipeline.cpp
  NAME HalideTestPipeline
  LINK_TO HalideTest)
