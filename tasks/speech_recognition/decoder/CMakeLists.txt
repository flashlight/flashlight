cmake_minimum_required(VERSION 3.5.1)

add_library(
  decoder
  INTERFACE
  )

target_sources(
  decoder
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/ConvLmModule.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Utils.cpp
  )

target_link_libraries(
  decoder
  INTERFACE
  common
  flashlight
  ${cereal_LIBRARIES}
  ${GLOG_LIBRARIES}
  )

target_include_directories(
  decoder
  INTERFACE
  ${GLOG_INCLUDE_DIRS}
  ${cereal_INCLUDE_DIRS}
  )
