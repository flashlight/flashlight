cmake_minimum_required(VERSION 3.5.1)

add_library(
  VisionNN
  INTERFACE
  )

set(
  NN_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/Transformer.h
  ${CMAKE_CURRENT_SOURCE_DIR}/PositionalEmbeddingSine.cpp
  )

target_sources(
  VisionNN
  INTERFACE
  $<BUILD_INTERFACE:${NN_SOURCES}>
  )
