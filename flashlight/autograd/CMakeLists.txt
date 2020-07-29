cmake_minimum_required(VERSION 3.5.1)

# ----------------------------- Autograd -----------------------------
add_library(
  Autograd
  INTERFACE
  )

set(
  AUTOGRAD_SOURCES
  ${CMAKE_SOURCE_DIR}/flashlight/autograd/Variable.cpp
  ${CMAKE_SOURCE_DIR}/flashlight/autograd/Functions.cpp
  ${CMAKE_SOURCE_DIR}/flashlight/autograd/Utils.cpp
  )

target_sources(
  Autograd
  INTERFACE
  $<BUILD_INTERFACE:${AUTOGRAD_SOURCES}>
  )

target_link_libraries(
  Autograd
  INTERFACE
  Common
)

target_include_directories(
  Autograd
  INTERFACE
  ${FLASHLIGHT_INCLUDE_DIR}
)

# ----------------------------- Autograd Backends -----------------------------
# CPU
if (FLASHLIGHT_USE_CPU)
  set(
    AUTOGRAD_CPU_SOURCES
    ${CMAKE_SOURCE_DIR}/flashlight/autograd/backend/cpu/Conv2D.cpp
    ${CMAKE_SOURCE_DIR}/flashlight/autograd/backend/cpu/Pool2D.cpp
    ${CMAKE_SOURCE_DIR}/flashlight/autograd/backend/cpu/RNN.cpp
    ${CMAKE_SOURCE_DIR}/flashlight/autograd/backend/cpu/BatchNorm.cpp # generic
    ${CMAKE_SOURCE_DIR}/flashlight/autograd/backend/cpu/MkldnnUtils.cpp # generic
    )
  
  target_sources(
    Autograd
    INTERFACE
    $<BUILD_INTERFACE:${AUTOGRAD_CPU_SOURCES}>
    )

  target_link_libraries(
    Autograd
    INTERFACE
    ${MKLDNN_LIBRARIES} # also contains MKL or mklml libraries
    )

  target_include_directories(
    Autograd
    INTERFACE
    ${CMAKE_SOURCE_DIR}
    ${MKLDNN_INCLUDE_DIR} # includes MKL headers if found
    )
endif ()


# CUDA
if (FLASHLIGHT_USE_CUDA)
  set(
    AUTOGRAD_CUDA_SOURCES
    ${CMAKE_SOURCE_DIR}/flashlight/autograd/backend/cuda/BatchNorm.cpp
    ${CMAKE_SOURCE_DIR}/flashlight/autograd/backend/cuda/Conv2D.cpp
    ${CMAKE_SOURCE_DIR}/flashlight/autograd/backend/cuda/CudnnUtils.h
    ${CMAKE_SOURCE_DIR}/flashlight/autograd/backend/cuda/CudnnUtils.cpp
    ${CMAKE_SOURCE_DIR}/flashlight/autograd/backend/cuda/Pool2D.cpp
    ${CMAKE_SOURCE_DIR}/flashlight/autograd/backend/cuda/RNN.cpp
    )
  
  target_sources(
    Autograd
    INTERFACE
    $<BUILD_INTERFACE:${AUTOGRAD_CUDA_SOURCES}>
    )

  target_link_libraries(
    Autograd
    INTERFACE
    ${CUDA_LIBRARIES}
    ${CUDNN_LIBRARIES}
    )

  target_include_directories(
    Autograd
    INTERFACE
    ${CMAKE_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${CUDNN_INCLUDE_DIRS}
    )

  target_compile_definitions(
    Autograd
    INTERFACE
    "-DNO_CUDNN_DESTROY_HANDLE"
    )
endif ()


# OpenCL
# TODO(jacobkahn): enable, with dependencies, when needed
if (FLASHLIGHT_USE_OPENCL)
  set(
    AUTOGRAD_OPENCL_SOURCES
    ${CMAKE_SOURCE_DIR}/flashlight/autograd/backend/opencl/Conv2D.cpp
    ${CMAKE_SOURCE_DIR}/flashlight/autograd/backend/opencl/Pool2D.cpp
    ${CMAKE_SOURCE_DIR}/flashlight/autograd/backend/opencl/RNN.cpp
    ${CMAKE_SOURCE_DIR}/flashlight/autograd/backend/generic/BatchNorm.cpp # generic
    )
  
  target_sources(
    Autograd
    INTERFACE
    $<BUILD_INTERFACE:${AUTOGRAD_OPENCL_SOURCES}>
    )

  target_link_libraries(
    Autograd
    INTERFACE
    ${OpenCL_LIBRARIES}
    )

  target_include_directories(
    Autograd
    INTERFACE
    ${CMAKE_SOURCE_DIR}
    ${OpenCL_INCLUDE_DIRS}
    )
endif ()
