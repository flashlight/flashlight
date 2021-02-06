cmake_minimum_required(VERSION 3.10)

# ----------------------------- Autograd -----------------------------
set(
  AUTOGRAD_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/Variable.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Functions.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Utils.cpp
  )

target_sources(
  flashlight
  PRIVATE
  ${AUTOGRAD_SOURCES}
  )

# ----------------------------- Autograd Backends -----------------------------
# CPU
if (FL_USE_CPU)
  find_package(DNNL 2.0 CONFIG REQUIRED)

  set(
    AUTOGRAD_CPU_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/backend/cpu/operators/AdvancedIndex.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/cpu/Conv2D.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/cpu/Pool2D.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/cpu/RNN.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/cpu/BatchNorm.cpp # generic
    ${CMAKE_CURRENT_LIST_DIR}/backend/cpu/DnnlUtils.cpp # generic
    )

  target_sources(
    flashlight
    PRIVATE
    ${AUTOGRAD_CPU_SOURCES}
    )

  target_link_libraries(
    flashlight
    PRIVATE
    DNNL::dnnl
    )
endif ()


# CUDA
if (FL_USE_CUDA)
  find_package(CUDNN 7.1) # CUDNN 7.1 works with CUDA 9.2
  if (CUDNN_FOUND)
    message(STATUS "CUDNN found (library: ${CUDNN_LIBRARIES} include: ${CUDNN_INCLUDE_DIRS})")
    setup_install_find_module(${CMAKE_MODULE_PATH}/FindCUDNN.cmake)
  else()
    message(STATUS "CUDNN not found")
    message(FATAL_ERROR "CUDNN required to build CUDA backend")
  endif()

  set(
    AUTOGRAD_CUDA_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/backend/cuda/operators/AdvancedIndex.cu
    ${CMAKE_CURRENT_LIST_DIR}/backend/cuda/BatchNorm.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/cuda/Conv2D.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/cuda/CudnnUtils.h
    ${CMAKE_CURRENT_LIST_DIR}/backend/cuda/CudnnUtils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/cuda/Pool2D.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/cuda/RNN.cpp
    )

  target_sources(
    flashlight
    PRIVATE
    ${AUTOGRAD_CUDA_SOURCES}
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
# Build OpenClSourceVerifierAndHeaderGen executable that generate c++
# header files from opencl source code and compile them code, using
# the opencl driver, to verify syntax at compile time.
# We find kernels under backend/opencl directory copy them as c++ headers to
# [build directory]/opencl_kernels
if (FL_USE_OPENCL)
  add_executable(
    opencl_syntax_and_header
    ${PROJECT_SOURCE_DIR}/cmake/OpenClSourceVerifierAndHeaderGen.cpp
    )

  target_link_libraries(
    opencl_syntax_and_header
    PUBLIC
    ${OpenCL_LIBRARIES}
    ArrayFire::${FL_AF_BACKEND}
    )

  target_include_directories(
    opencl_syntax_and_header
    PUBLIC
    ${OpenCL_INCLUDE_DIRS}
    )


  file(GLOB kernel_src ${CMAKE_CURRENT_LIST_DIR}/backend/opencl/*.cl)
  set(OPENCL_KERNEL_DIR "opencl_kernels")
  include(OpenClSourceVerifierAndHeaderGen)

  opencl_syntax_check_and_header_gen(
      SOURCES ${kernel_src}
      VARNAME kernel_files
      EXTENSION "h"  # extention to the header file contining opencl source.
      OUTPUT_DIR ${OPENCL_KERNEL_DIR}
      TARGETS cl_kernel_targets
      NAMESPACE "opencl" # namespace that wraps the string in the file.
      )

  set(opencl_compile_definitions
          CL_TARGET_OPENCL_VERSION=120
          CL_HPP_TARGET_OPENCL_VERSION=120
          CL_HPP_MINIMUM_OPENCL_VERSION=120
          CL_HPP_ENABLE_EXCEPTIONS
          CL_USE_DEPRECATED_OPENCL_1_2_APIS)

  set(
    AUTOGRAD_OPENCL_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/backend/cpu/operators/AdvancedIndex.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/opencl/Conv2D.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/opencl/Pool2D.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/opencl/RNN.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/opencl/BatchNorm.cpp # generic
    )

  target_sources(
    flashlight
    PRIVATE
    ${AUTOGRAD_OPENCL_SOURCES}
    ${kernel_files}
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

  target_include_directories(
    flashlight
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
  )

  target_compile_definitions(
    flashlight
    PRIVATE
    ${opencl_compile_definitions}
    )

  add_dependencies(flashlight ${cl_kernel_targets})
endif ()
