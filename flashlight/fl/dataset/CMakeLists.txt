cmake_minimum_required(VERSION 3.5.1)

set(
  DATASET_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/BatchDataset.cpp
  ${CMAKE_CURRENT_LIST_DIR}/BatchDataset.h
  ${CMAKE_CURRENT_LIST_DIR}/BlobDataset.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ConcatDataset.cpp
  ${CMAKE_CURRENT_LIST_DIR}/DatasetIterator.h
  ${CMAKE_CURRENT_LIST_DIR}/Utils.cpp
  ${CMAKE_CURRENT_LIST_DIR}/FileBlobDataset.cpp
  ${CMAKE_CURRENT_LIST_DIR}/MemoryBlobDataset.cpp
  ${CMAKE_CURRENT_LIST_DIR}/MergeDataset.cpp
  ${CMAKE_CURRENT_LIST_DIR}/PrefetchDataset.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ResampleDataset.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ShuffleDataset.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TensorDataset.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TransformDataset.cpp
  )

target_sources(
  flashlight
  PUBLIC
  $<BUILD_INTERFACE:${DATASET_SOURCES}>
  )
