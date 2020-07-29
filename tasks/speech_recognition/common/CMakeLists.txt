cmake_minimum_required(VERSION 3.5.1)

add_library(
  common
  INTERFACE
  )

target_sources(
  common
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/Defines.cpp
  )

target_link_libraries(
  common
  INTERFACE
  flashlight
  ${GFLAGS_LIBRARIES}
  )

target_include_directories(
  common
  INTERFACE
  ${CMAKE_SOURCE_DIR}
  ${GFLAGS_INCLUDE_DIRS}
  )
