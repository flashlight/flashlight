cmake_minimum_required(VERSION 3.10)

option(FL_USE_ARRAYFIRE "Build ArrayFire tensor backend" ON)

if (FL_USE_ARRAYFIRE)
  include(${CMAKE_CURRENT_LIST_DIR}/backend/af/CMakeLists.txt)
endif()

target_compile_definitions(
  flashlight
  PUBLIC
  FL_USE_ARRAYFIRE=$<BOOL:${FL_USE_ARRAYFIRE}>
)

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Compute.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Index.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Init.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Random.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Shape.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Stream.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TensorBackend.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TensorBase.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TensorAdapter.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TensorExtension.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Types.cpp
)

if (FL_USE_CUDA)
  target_sources(
    flashlight
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/CUDAStream.cpp
    ${CMAKE_CURRENT_LIST_DIR}/CUDAUtils.cpp

    )
  if (FL_BUILD_PROFILING)
    target_sources(flashlight PRIVATE ${CMAKE_CURRENT_LIST_DIR}/CUDAProfile.cpp)
  endif()
endif()
