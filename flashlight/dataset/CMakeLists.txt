cmake_minimum_required(VERSION 3.5.1)

add_library(
  Dataset
  INTERFACE
  )

set(
  DATASET_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/dataset/BatchDataset.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/dataset/BatchDataset.h
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/dataset/BlobDataset.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/dataset/ConcatDataset.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/dataset/DatasetIterator.h
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/dataset/Utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/dataset/FileBlobDataset.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/dataset/ImageDataset.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/dataset/MergeDataset.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/dataset/PrefetchDataset.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/dataset/ResampleDataset.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/dataset/ShuffleDataset.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/dataset/TensorDataset.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/dataset/TransformDataset.cpp
  )

target_sources(
  Dataset
  INTERFACE
  $<BUILD_INTERFACE:${DATASET_SOURCES}>
  )
