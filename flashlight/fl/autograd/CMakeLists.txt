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
if (FL_USE_OPENCL)
  include(OpenCLUtils)

  # Scan for OpenCL kernels under the backend/opencl directory and copy them
  # as c++ headers to [build directory]/opencl_kernels
  opencl_add_kernels_in_dir_to_target(
    DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/backend/opencl
    TARGET flashlight
    )

  set(
    AUTOGRAD_OPENCL_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/backend/cpu/operators/AdvancedIndex.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/opencl/Conv2D.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/opencl/Pool2D.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/opencl/RNN.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/opencl/BatchNorm.cpp
    )

  target_sources(
    flashlight
    PRIVATE
    ${AUTOGRAD_OPENCL_SOURCES}
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


message("-------------------------- FL_USE_MIOPEN=${FL_USE_MIOPEN}")
# MIOPEN
if (FL_USE_MIOPEN)
  find_package(MIOPEN)
  if (MIOPEN_FOUND)
    message(STATUS "MIOPEN found (library: ${MIOPEN_LIBRARIES} include: ${MIOPEN_INCLUDE_DIRS})")
    setup_install_find_module(${CMAKE_MODULE_PATH}/FindMIOPEN.cmake)
  else()
    message(STATUS "MIOPEN not found")
    message(FATAL_ERROR "MIOPEN required to build CUDA backend")
  endif()

  set(
    AUTOGRAD_MIOPEN_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/backend/cpu/operators/AdvancedIndex.cpp
    # ${CMAKE_CURRENT_LIST_DIR}/backend/miopen/BatchNorm.cpp # unsupported by AWS GPU gfx1011
    ${CMAKE_CURRENT_LIST_DIR}/backend/opencl/BatchNorm.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/miopen/Conv2D.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/miopen/MiOpenUtils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/miopen/Pool2D.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/miopen/RNN.cpp
    )

  target_sources(
    flashlight
    PRIVATE
    ${AUTOGRAD_MIOPEN_SOURCES}
    )

  target_link_libraries(
    flashlight
    PUBLIC
    ${MIOPEN_LIBRARIES}
    ${OpenCL_LIBRARIES}
    )

  target_include_directories(
    flashlight
    PUBLIC
    ${MIOPEN_INCLUDE_DIRS}
    ${OpenCL_INCLUDE_DIRS}
    )

  target_compile_definitions(
    flashlight
    PRIVATE
    ${opencl_compile_definitions}
    )
endif ()
