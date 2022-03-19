cmake_minimum_required(VERSION 3.10)

option(FL_USE_CUDNN "Build ArrayFire tensor backend" OFF)
option(FL_USE_ONEDNN "Build ArrayFire tensor backend" OFF)

# Try to find cuDNN
find_package(CUDNN 7.1) # CUDNN 7.1 works with CUDA 9.2
if (NOT CUDNN_FOUND AND FL_USE_CUDNN)
  message(FATAL_ERROR
    "cuDNN required to build cuDNN backend: cuDNN was not found")
endif()
if (CUDNN_FOUND AND FL_USE_CUDA)
  message(STATUS
    "CUDNN found (library: ${CUDNN_LIBRARIES} include: ${CUDNN_INCLUDE_DIRS})")
  set(FL_USE_CUDNN ON FORCE)
  setup_install_find_module(${CMAKE_MODULE_PATH}/FindCUDNN.cmake)
endif()

# Try to find oneDNN
find_package(DNNL 2.0 CONFIG)
if (NOT DNNL_FOUND AND FL_USE_ONEDNN)
  message(FATAL_ERROR
    "oneDNN required to build oneDNN backend: oneDNN was not found")
endif()
if (DNNL_FOUND)
  set(FL_USE_ONEDNN ON FORCE)
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
