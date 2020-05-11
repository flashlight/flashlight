cmake_minimum_required(VERSION 3.5.1)

add_library(
  Meter
  INTERFACE
  )

set(
  METER_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/meter/AverageValueMeter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/meter/CountMeter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/meter/EditDistanceMeter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/meter/FrameErrorMeter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/meter/MSEMeter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/meter/TimeMeter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/meter/TopKMeter.cpp
  )

target_sources(
  Meter
  INTERFACE
  $<BUILD_INTERFACE:${METER_SOURCES}>
  )

