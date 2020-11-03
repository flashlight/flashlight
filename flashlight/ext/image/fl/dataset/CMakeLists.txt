cmake_minimum_required(VERSION 3.5.1)

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/DistributedDataset.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Jpeg.cpp
  ${CMAKE_CURRENT_LIST_DIR}/LoaderDataset.h
)
