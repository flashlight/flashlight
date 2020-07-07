cmake_minimum_required(VERSION 3.5.1)

add_library(
  asr-runtime
  INTERFACE
  )

target_sources(
  asr-runtime
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/Logger.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SpeechStatMeter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Optimizer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Helpers.cpp
  )

target_link_libraries(
  asr-runtime
  INTERFACE
  asr-common
  asr-data
  flashlight
  ${GLOG_LIBRARIES}
  ${cereal_LIBRARIES}
  )

target_include_directories(
  asr-runtime
  INTERFACE
  ${GLOG_INCLUDE_DIRS}
  ${cereal_INCLUDE_DIRS}
  )
