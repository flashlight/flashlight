cmake_minimum_required(VERSION 3.10)

option(FL_USE_ARRAYFIRE "Build ArrayFire tensor backend" ON)

if (FL_USE_ARRAYFIRE)
  include(${CMAKE_CURRENT_LIST_DIR}/backend/af/CMakeLists.txt)
endif()

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Shape.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TensorBase.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Types.cpp
)
