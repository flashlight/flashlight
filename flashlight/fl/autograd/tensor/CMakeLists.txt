cmake_minimum_required(VERSION 3.10)

option(FL_USE_CUDNN "Build ArrayFire tensor backend" OFF)
option(FL_USE_ONEDNN "Build ArrayFire tensor backend" OFF)

if (FL_USE_CUDNN)
  include(${CMAKE_CURRENT_LIST_DIR}/backend/cudnn/CMakeLists.txt)
endif()

if (FL_USE_ONEDNN)
  include(${CMAKE_CURRENT_LIST_DIR}/backend/onednn/CMakeLists.txt)
endif()

target_compile_definitions(
  flashlight
  PUBLIC
  FL_USE_CUDNN=$<BOOL:${FL_USE_CUDNN}>
  FL_USE_ONEDNN=$<BOOL:${FL_USE_ONEDNN}>
)

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/AutogradOps.cpp
)
