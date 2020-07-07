cmake_minimum_required(VERSION 3.5.1)

add_library(
  lm-data
  INTERFACE
  )

target_sources(
  lm-data
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/BlobDataset.cpp
  )

target_link_libraries(
  lm-data
  INTERFACE
  flashlight
  ${GFLAGS_LIBRARIES}
  )

target_include_directories(
  lm-data
  INTERFACE
  ${CMAKE_SOURCE_DIR}
  ${GFLAGS_INCLUDE_DIRS}
  )
