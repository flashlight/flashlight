cmake_minimum_required(VERSION 3.10)

target_sources(
  fl-libraries
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/String.cpp
  ${CMAKE_CURRENT_LIST_DIR}/System.cpp
  )
