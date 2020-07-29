cmake_minimum_required(VERSION 3.5.1)

add_library(
  libraries-language-dictionary
  INTERFACE
  )

target_sources(
  libraries-language-dictionary
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/Dictionary.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Utils.cpp
  )
