cmake_minimum_required(VERSION 3.10)

# CUDA
find_package(CUDNN 7.1) # CUDNN 7.1 works with CUDA 9.2
if (CUDNN_FOUND)
  message(STATUS "CUDNN found (library: ${CUDNN_LIBRARIES} include: ${CUDNN_INCLUDE_DIRS})")
  setup_install_find_module(${CMAKE_MODULE_PATH}/FindCUDNN.cmake)
else()
  # TODO: just don't build the backend
  message(STATUS "CUDNN not found")
  message(FATAL_ERROR "CUDNN required to build CUDNN backend")
endif()

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/CudnnAutogradExtension.cpp
  ${CMAKE_CURRENT_LIST_DIR}/BatchNorm.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Conv2D.cpp
  ${CMAKE_CURRENT_LIST_DIR}/CudnnUtils.h
  ${CMAKE_CURRENT_LIST_DIR}/CudnnUtils.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Pool2D.cpp
  ${CMAKE_CURRENT_LIST_DIR}/RNN.cpp
  )

target_link_libraries(
  flashlight
  PUBLIC
  ${CUDNN_LIBRARIES}
  )

target_include_directories(
  flashlight
  PUBLIC
  ${CUDNN_INCLUDE_DIRS}
  )

target_compile_definitions(
  flashlight
  PUBLIC
  "-DNO_CUDNN_DESTROY_HANDLE"
  )
