cmake_minimum_required(VERSION 3.16)

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
  ${CUDNN_LIBRARY_PATH}
  )

target_include_directories(
  flashlight
  PUBLIC
  ${CUDNN_INCLUDE_PATH}
  )

target_compile_definitions(
  flashlight
  PUBLIC
  "-DNO_CUDNN_DESTROY_HANDLE"
  "-DNO_CUDNN_DESTROY_STREAM"
  )
