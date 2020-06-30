cmake_minimum_required(VERSION 3.5.1)

add_library(
  extension-common
  INTERFACE
  )

target_sources(
  extension-common
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/SequentialBuilder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Utils.cpp
  )
