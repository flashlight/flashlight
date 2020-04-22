cmake_minimum_required(VERSION 3.5.1)

add_library(
  VisionDatasets
  INTERFACE
  )

set(
  DATASET_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/ImageDataset.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/VisionDataset.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ImagenetUtils.cpp
  )

target_sources(
  VisionDatasets
  INTERFACE
  $<BUILD_INTERFACE:${DATASET_SOURCES}>
  )
