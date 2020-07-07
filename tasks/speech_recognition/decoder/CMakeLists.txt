cmake_minimum_required(VERSION 3.5.1)

add_library(
  asr-decoder
  INTERFACE
  )

target_sources(
  asr-decoder
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/ConvLmModule.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Utils.cpp
  )

target_link_libraries(
  asr-decoder
  INTERFACE
  asr-common
  flashlight
  ${cereal_LIBRARIES}
  ${GLOG_LIBRARIES}
  )

target_include_directories(
  asr-decoder
  INTERFACE
  ${GLOG_INCLUDE_DIRS}
  ${cereal_INCLUDE_DIRS}
  )
