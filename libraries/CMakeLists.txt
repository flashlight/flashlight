cmake_minimum_required(VERSION 3.5.1)

# ------------------------- Library -------------------------

add_library(
  fl-libraries
  ""
  )
set(INSTALLABLE_TARGETS ${INSTALLABLE_TARGETS} fl-libraries)

target_include_directories(
  fl-libraries
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/..> # for #include "flashlight/libraries"
  )

target_compile_definitions(
  fl-libraries
  PUBLIC
  ${FL_COMPILE_DEFINITIONS}
  )

# --------------------------- Components ---------------------------

# sequence
include(${CMAKE_CURRENT_LIST_DIR}/sequence/CMakeLists.txt)

# audio
include(${CMAKE_CURRENT_LIST_DIR}/audio/CMakeLists.txt)

# common
include(${CMAKE_CURRENT_LIST_DIR}/common/CMakeLists.txt)

# text
include(${CMAKE_CURRENT_LIST_DIR}/text/CMakeLists.txt)

# ---------------------- CUDA Library -----------------------

# Accumulate CUDA sources after including library components because
# target_sources doesn't work with CUDA libraries as expected with
# old versions of CMake
set(FL_LIBRARY_CUDA_SOURCES
  ${FL_LIBRARIES_SEQUENCE_CUDA_SOURCES}
  )

# A library containing CUDA targets within libraries that may need to be linked
# in. Only build a CUDA lib of libraries that require CUDA kernels are built
if (FL_LIBRARIES_USE_CUDA)
  include(${CMAKE_MODULE_PATH}/BuildCUB.cmake)
  include(${CMAKE_MODULE_PATH}/CUDAUtils.cmake)

  set_cuda_cxx_compile_flags()
  set_cuda_arch_nvcc_flags()

  # hacky: add -fPIC to nvcc flags if needed
  if (FL_LIBRARIES_BUILD_FOR_PYTHON OR CMAKE_POSITION_INDEPENDENT_CODE)
    cuda_enable_position_independent_code()
  endif ()
  
  cuda_include_directories(
    ${CUB_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR} # includes are relative to the project path
    )

  # Qualify name because CUDA target is public
  cuda_add_library(
    fl-libraries-cuda
    ${FL_LIBRARY_CUDA_SOURCES}
    )

  set(INSTALLABLE_TARGETS ${INSTALLABLE_TARGETS} fl-libraries-cuda)
  add_dependencies(fl-libraries-cuda CUB)
  
  cuda_include_directories(
    ${CUB_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR} # TODO(jacobkahn): fix me
    )

  target_link_libraries(
    fl-libraries-cuda
    PUBLIC
    ${CUDA_LIBRARIES}
    )

  # link to fl-libraries
  target_link_libraries(
    fl-libraries
    PUBLIC
    fl-libraries-cuda
    )

  target_include_directories(
    fl-libraries
    PUBLIC
    $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
    )
endif()

# --------------------------- Tests ---------------------------

if (FL_BUILD_TESTS)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test)
endif ()
