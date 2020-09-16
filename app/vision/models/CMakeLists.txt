cmake_minimum_required(VERSION 3.5.1)

add_library(
  VisionModels
  INTERFACE
  )

add_dependencies(VisionModels cereal)

set(
  VISION_MODELS_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/Resnet.cpp
  )

target_sources(
  VisionModels
  INTERFACE
  $<BUILD_INTERFACE:${VISION_MODELS_SOURCES}>
  )

target_link_libraries(
  VisionModels
  INTERFACE
  NN # flashlight Modules
  )
