cmake_minimum_required(VERSION 3.10)

option(FL_USE_CUDNN "Build ArrayFire tensor backend" ON)

if (FL_USE_CUDNN)
  include(${CMAKE_CURRENT_LIST_DIR}/backend/cudnn/CMakeLists.txt)
endif()

target_compile_definitions(
  flashlight
  PUBLIC
  FL_USE_CUDNN=$<BOOL:${FL_USE_CUDNN}>
)

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/AutogradOps.cpp
)
