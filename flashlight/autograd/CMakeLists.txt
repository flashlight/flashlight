cmake_minimum_required(VERSION 3.5.1)

# ----------------------------- Autograd -----------------------------
set(
  AUTOGRAD_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/Variable.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Functions.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Utils.cpp
  )

target_sources(
  flashlight
  PUBLIC
  $<BUILD_INTERFACE:${AUTOGRAD_SOURCES}>
  )

# ----------------------------- Autograd Backends -----------------------------
# CPU
if (FLASHLIGHT_USE_CPU)
  set(
    AUTOGRAD_CPU_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/backend/cpu/Conv2D.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/cpu/Pool2D.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/cpu/RNN.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/cpu/BatchNorm.cpp # generic
    ${CMAKE_CURRENT_LIST_DIR}/backend/cpu/MkldnnUtils.cpp # generic
    )
  
  target_sources(
    flashlight
    PUBLIC
    $<BUILD_INTERFACE:${AUTOGRAD_CPU_SOURCES}>
    )

  target_link_libraries(
    flashlight
    PUBLIC
    ${MKLDNN_LIBRARIES} # also contains MKL or mklml libraries
    )

  target_include_directories(
    flashlight
    PUBLIC
    ${MKLDNN_INCLUDE_DIR} # includes MKL headers if found
    )
endif ()


# CUDA
if (FLASHLIGHT_USE_CUDA)  
  find_package(CUDNN 7.1 QUIET) # CUDNN 7.1 works with CUDA 9.2
  if (CUDNN_FOUND)
    message(STATUS "CUDNN found (library: ${CUDNN_LIBRARIES} include: ${CUDNN_INCLUDE_DIRS})")
  else()
    message(STATUS "CUDNN not found")
    message(FATAL_ERROR "CUDNN required to build CUDA backend")
  endif()
  
  set(
    AUTOGRAD_CUDA_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/backend/cuda/BatchNorm.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/cuda/Conv2D.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/cuda/CudnnUtils.h
    ${CMAKE_CURRENT_LIST_DIR}/backend/cuda/CudnnUtils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/cuda/Pool2D.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/cuda/RNN.cpp
    )
  
  target_sources(
    flashlight
    PUBLIC
    $<BUILD_INTERFACE:${AUTOGRAD_CUDA_SOURCES}>
    )

  target_link_libraries(
    flashlight
    PUBLIC
    ${CUDA_LIBRARIES}
    ${CUDNN_LIBRARIES}
    )

  target_include_directories(
    flashlight
    PUBLIC
    ${CUDA_INCLUDE_DIRS}
    ${CUDNN_INCLUDE_DIRS}
    )

  target_compile_definitions(
    flashlight
    PUBLIC
    "-DNO_CUDNN_DESTROY_HANDLE"
    )
endif ()


# OpenCL
# TODO(jacobkahn): enable, with dependencies, when needed
if (FLASHLIGHT_USE_OPENCL)
  set(
    AUTOGRAD_OPENCL_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/backend/opencl/Conv2D.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/opencl/Pool2D.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/opencl/RNN.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/generic/BatchNorm.cpp # generic
    )
  
  target_sources(
    flashlight
    PUBLIC
    $<BUILD_INTERFACE:${AUTOGRAD_OPENCL_SOURCES}>
    )

  target_link_libraries(
    flashlight
    PUBLIC
    ${OpenCL_LIBRARIES}
    )

  target_include_directories(
    flashlight
    PUBLIC
    ${OpenCL_INCLUDE_DIRS}
    )
endif ()
