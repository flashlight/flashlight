cmake_minimum_required(VERSION 3.5.1)

add_library(
  libraries-common
  INTERFACE
  )

target_sources(
  libraries-common
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/String.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/System.cpp
  )
