cmake_minimum_required(VERSION 3.16)

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Variable.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Functions.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Utils.cpp
  )

include(${CMAKE_CURRENT_LIST_DIR}/tensor/CMakeLists.txt)
