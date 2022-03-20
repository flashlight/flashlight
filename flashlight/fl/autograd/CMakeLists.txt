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

include(${CMAKE_CURRENT_LIST_DIR}/tensor/CMakeLists.txt)

# ----------------------------- Autograd Backends -----------------------------
# TODO: move to files in autograd/tensor/backend once AutogradExtensions for these
# implementations are complete

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
