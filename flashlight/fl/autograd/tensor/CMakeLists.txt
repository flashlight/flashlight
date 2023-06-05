cmake_minimum_required(VERSION 3.16)

option(FL_USE_CUDNN "Build with cuDNN support" OFF)

find_package(CUDNN)
if (NOT CUDNN_FOUND AND FL_USE_CUDNN)
  message(FATAL_ERROR
    "cuDNN required to build cuDNN backend: cuDNN was not found")
endif()
if (CUDNN_FOUND AND FL_USE_CUDA)
  message(STATUS
    "CUDNN found (library: ${CUDNN_LIBRARIES} include: ${CUDNN_INCLUDE_DIRS})")
  set(FL_USE_CUDNN ON)
  setup_install_find_module(${PROJECT_SOURCE_DIR}/cmake/FindCUDNN.cmake)
endif()

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
