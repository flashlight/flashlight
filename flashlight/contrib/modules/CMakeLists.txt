cmake_minimum_required(VERSION 3.5.1)

add_library(
  ContribModules
  INTERFACE
  )

add_dependencies(ContribModules cereal)

set(
  CONTRIB_MODULES_SOURCES
  ${FL_CONTRIB_ROOT_PATH}/modules/AsymmetricConv1D.cpp
  ${FL_CONTRIB_ROOT_PATH}/modules/PositionEmbedding.cpp
  ${FL_CONTRIB_ROOT_PATH}/modules/Residual.cpp
  ${FL_CONTRIB_ROOT_PATH}/modules/Transformer.cpp
  ${FL_CONTRIB_ROOT_PATH}/modules/TDSBlock.cpp
  ${FL_CONTRIB_ROOT_PATH}/modules/SpecAugment.cpp
  )

target_sources(
  ContribModules
  INTERFACE
  $<BUILD_INTERFACE:${CONTRIB_MODULES_SOURCES}>
  )

target_link_libraries(
  ContribModules
  INTERFACE
  NN # flashlight Modules
  )
