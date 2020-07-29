cmake_minimum_required(VERSION 3.5.1)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ---------------------- Configuration ----------------------

option(FL_LIBRARIES_USE_CUDA "Use CUDA in libraries-only build" ON)
option(FL_LIBRARIES_USE_KENLM "Use KenLM in libraries-only build" ON)
option(FL_LIBRARIES_USE_MKL "Use MKL in libraries-only build" ON)
option(FL_BUILD_FOR_PYTHON "Build Python bindings" OFF)

set(FL_COMPILE_DEFINITIONS
  $<$<BOOL:${FL_LIBRARIES_USE_CUDA}>:FL_LIBRARIES_USE_CUDA>
  $<$<BOOL:${FL_LIBRARIES_USE_KENLM}>:FL_LIBRARIES_USE_KENLM>
  $<$<BOOL:${FL_LIBRARIES_USE_MKL}>:FL_LIBRARIES_USE_MKL>
  )

# ------------------------- Library -------------------------

add_library(
  fl-libraries
  ""
  )

target_include_directories(
  fl-libraries
  PUBLIC
  ${PROJECT_SOURCE_DIR}
  )

target_compile_definitions(
  fl-libraries
  PUBLIC
  ${FL_LIBRARIES_COMPILE_DEFINITIONS}
  )

# ---------------------- CUDA Library -----------------------

# A library containing CUDA targets within libraries that may need to be linked
# in. Only build a CUDA lib of libraries that require CUDA kernels are built
if (FL_LIBRARIES_USE_CUDA)
  include(${CMAKE_MODULE_PATH}/BuildCUB.cmake)
  include(${CMAKE_MODULE_PATH}/CUDAUtils.cmake)

  set_cuda_cxx_compile_flags()
  set_cuda_arch_nvcc_flags()

  # hacky: add -fPIC to nvcc flags if needed
  if (FL_BUILD_FOR_PYTHON OR CMAKE_POSITION_INDEPENDENT_CODE)
    cuda_enable_position_independent_code()
  endif ()
  
  cuda_include_directories(
    ${CUB_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src
    )

  # Qualify name because CUDA target is public
  cuda_add_library(
    fl-libraries-cuda
    ""
    )

  add_dependencies(fl-libraries-cuda CUB)
  
  cuda_include_directories(
    ${CUB_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}
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
    ${CUDA_INCLUDE_DIRS}
    )

endif()

# --------------------------- Components ---------------------------

# sequence
include(${CMAKE_CURRENT_SOURCE_DIR}/sequence/CMakeLists.txt)

# audio
include(${CMAKE_CURRENT_SOURCE_DIR}/audio/CMakeLists.txt)

# common
include(${CMAKE_CURRENT_SOURCE_DIR}/common/CMakeLists.txt)

# language
include(${CMAKE_CURRENT_SOURCE_DIR}/language/CMakeLists.txt)

# --------------------------- Tests ---------------------------

# Build tests
option(FL_BUILD_TESTS "Build tests for flashlight" ON)
if (FL_BUILD_TESTS)
  enable_testing()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)
endif ()
