cmake_minimum_required(VERSION 3.10)

if (FL_USE_CUDA)
set(
  DATASET_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/backend/cuda/Sample.cpp
  ${CMAKE_CURRENT_LIST_DIR}/BatchDataset.cpp
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
else()
set(
  DATASET_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/backend/cpu/Sample.cpp
  ${CMAKE_CURRENT_LIST_DIR}/BatchDataset.cpp
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
endif()


target_sources(
  flashlight
  PRIVATE
  ${DATASET_SOURCES}
  )
